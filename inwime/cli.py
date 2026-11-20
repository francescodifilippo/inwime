"""The `inwime` command."""
import argparse
import json
import os
import sys

from . import __version__, catalog, doctor, storage


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

    m = commands.add_parser('models', help='Hugging Face catalog: search and inspect models')
    models = m.add_subparsers(dest='models_command', metavar='MODELS_COMMAND')
    i = models.add_parser('info', help='describe one repository at one revision (no weights downloaded)')
    i.add_argument('spec', metavar='REPO[@REVISION]', help='e.g. Qwen/Qwen2.5-7B-Instruct@<full sha>')
    i.add_argument('--json', action='store_true', help='machine-readable output (schema inwime-model-info/1)')
    s = models.add_parser('search', help='search the whole Hugging Face catalog (no whitelist)')
    s.add_argument('text', nargs='?', help='text contained in the repository id')
    s.add_argument('--author', help='user or organization')
    s.add_argument('--task', help='pipeline tag, e.g. text-generation')
    s.add_argument('--family', help='architecture family tag, e.g. qwen2, llama')
    s.add_argument('--license', dest='licence', help='licence id, e.g. apache-2.0')
    s.add_argument('--min-params', help='minimum parameter count, e.g. 500M, 7B')
    s.add_argument('--max-params', help='maximum parameter count, e.g. 14B')
    s.add_argument('--sort', choices=tuple(catalog.SORTS),
                   default='downloads', help='popularity or date order, descending (default: downloads)')
    s.add_argument('--limit', type=int, default=20,
                   help='results per page, 1-%d (default: 20)' % catalog.MAX_PAGE_SIZE)
    s.add_argument('--page', type=int, default=1, help='page number, from 1')
    s.add_argument('--json', action='store_true', help='machine-readable output (schema inwime-model-search/1)')
    return parser


def models_main(parser, args):
    if args.models_command is None:
        parser.parse_args(['models', '--help'])
    try:
        if args.models_command == 'info':
            result = catalog.info(args.spec)
        else:
            result = catalog.search(args.text, author=args.author, task=args.task, family=args.family,
                                    licence=args.licence, min_params=args.min_params,
                                    max_params=args.max_params, sort=args.sort, limit=args.limit,
                                    page=args.page)
    except catalog.CatalogError as error:
        sys.stderr.write('inwime: %s\n' % error)
        return error.exit_status
    if args.json:
        json.dump(result, sys.stdout, indent=1)
        sys.stdout.write('\n')
    else:
        render = catalog.render_info if args.models_command == 'info' else catalog.render_search
        sys.stdout.write(render(result))
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'models':
        return models_main(parser, args)
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
