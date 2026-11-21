"""Test-only network guard, loaded by every `inwime` subprocess the product tests start.

Any name resolution or connection to a non-loopback address fails immediately and is
appended to the file named by INWIME_NETGUARD_LOG, so a test can prove that no request left
the machine even if the code under test swallowed the exception. The exception is not an
OSError on purpose: HTTP clients would turn it into a retryable connection error.
"""
import os
import socket

LOOPBACK = {'127.0.0.1', '::1', 'localhost'}


class NetworkGuardViolation(RuntimeError):
    pass


def _violation(what):
    log = os.environ.get('INWIME_NETGUARD_LOG')
    if log:
        with open(log, 'a') as f:
            f.write(what+'\n')
    raise NetworkGuardViolation('INWIME-NETGUARD blocked ' + what)


_getaddrinfo = socket.getaddrinfo
_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex


def getaddrinfo(host, *args, **kwargs):
    name = host.decode() if isinstance(host, bytes) else host
    if name is not None and name not in LOOPBACK:
        _violation('name resolution of %r' % (name,))
    return _getaddrinfo(host, *args, **kwargs)


def _check(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in LOOPBACK:
        _violation('connection to %r' % (address,))


def connect(self, address):
    _check(self, address)
    return _connect(self, address)


def connect_ex(self, address):
    _check(self, address)
    return _connect_ex(self, address)


socket.getaddrinfo = getaddrinfo
socket.socket.connect = connect
socket.socket.connect_ex = connect_ex


# Test-only fake of the physical free space seen by `inwime models install` (fakestorage.py).
if os.environ.get('INWIME_FAKE_FREE_BYTES'):
    import fakestorage
    fakestorage.install()
