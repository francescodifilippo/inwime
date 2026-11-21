"""Test-only stand-in for the physical-space probe of `inwime models install`.

Loaded by sitecustomize.py when INWIME_FAKE_FREE_BYTES is set: a comma-separated list of the
physical free bytes successive measures report (the last one repeats); `unknown` reports an
undetectable backing, as the real probe does under WSL without the Windows host query. Each
measure is appended to INWIME_FAKE_FREE_LOG when set, so a test can count the checks.
"""
import os


class FakeSpaceProbe:
    def __init__(self, path, margin_bytes):
        self.path, self.margin_bytes = str(path), margin_bytes
        self.values = os.environ['INWIME_FAKE_FREE_BYTES'].split(',')

    def measure(self):
        value = self.values.pop(0) if len(self.values) > 1 else self.values[0]
        log = os.environ.get('INWIME_FAKE_FREE_LOG')
        if log:
            with open(log, 'a') as f:
                f.write(value+'\n')
        free = None if value == 'unknown' else int(value)
        return dict(path=self.path, exists=os.path.exists(self.path), writable=True,
                    mount_point='/fake', fstype='fakefs', source='fake', backing='local' if free is not None
                    else 'unknown', host_volume=None, vhdx=None, linux_free_bytes=free,
                    physical_free_bytes=free, margin_bytes=self.margin_bytes,
                    usable_bytes=None if free is None else free-self.margin_bytes,
                    notes=[] if free is not None else ['fake: backing unknown'],
                    status='ok' if free is not None else 'unknown')


def install():
    import inwime.installer
    inwime.installer.SpaceProbe = FakeSpaceProbe
