"""The `inwime` command."""
import argparse
import json
import os
import sys

from . import __version__, doctor, storage


def build_parser():
    parser = argparse.ArgumentParser(prog='inwime', description='InWiMe: out-of-core local inference.')
    parser.add_argument('--version', action='version', version='inwime '+__version__)
    commands = parser.add_subparsers(dest='command', metavar='COMMAND')
    d = commands.add_parser('doctor', help='non-destructive audit of hardware, memory, storage and kernel')
    d.add_argument('--json', action='store_true', help='machine-readable report (schema inwime-doctor/1)')
    d.add_argument('--path', action='append', default=None, metavar='DIR',
                   help='storage location to assess (repeatable; default: Hugging Face hub cache)')
    d.add_argument('--host-margin-gib', type=float, default=storage.DEFAULT_MARGIN_BYTES/storage.GIB,
                   help='free space that must remain on the physical volume (default: 30)')
    d.add_argument('--no-host-query', action='store_true',
                   help='under WSL, do not query the Windows host through PowerShell')
    d.add_argument('--show-paths', action='store_true', help='do not redact user names in paths')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != 'doctor':
        parser.print_help()
        return 2
    if args.host_margin_gib < 0:
        parser.error('--host-margin-gib must not be negative')
    result = doctor.report(paths=args.path, host_query=not args.no_host_query,
                           margin_bytes=int(args.host_margin_gib*storage.GIB))
    if not args.show_paths:
        result = doctor.redact(result, home=os.path.expanduser('~'))
    if args.json:
        json.dump(result, sys.stdout, indent=1)
        sys.stdout.write('\n')
    else:
        sys.stdout.write(doctor.render(result))
    return 1 if result['status'] == 'fail' else 0
