#!/usr/bin/env python

from __future__ import division, print_function

import os
import subprocess
import sys

from setuptools import setup, find_packages

__author__ = "Gary Lai"

MAJOR = 1
MINOR = 0
MICRO = 0
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# Return the git revision as a string
def get_dev_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        # out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        out = _minimal_ext_cmd(['git', 'log', '-1', '--pretty=format:%ct'])
        DEV_VERSION = out.strip().decode()
    except OSError:
        DEV_VERSION = "Unknown"

    return DEV_VERSION


def get_version_info(filename):
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        DEV_VERSION = get_dev_version()
    elif os.path.exists(filename):
        # must be a source distribution, use existing version file
        try:
            from bert_embedding.version import dev_version as DEV_VERSION
        except ImportError:
            raise ImportError("Unable to import dev_version. Try removing "
                              "bert_embedding/version.py and the build directory "
                              "before building.")
    else:
        DEV_VERSION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev' + DEV_VERSION

    return FULLVERSION, DEV_VERSION


def write_version_py(filename):
    cnt = """
# THIS FILE IS GENERATED FROM SCIPY SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
dev_version = '%(dev_version)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, DEV_VERSION = get_version_info(filename)

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'dev_version': DEV_VERSION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


VERSION_PY = 'bert_embedding/version.py'

if sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version >= 3.4 required.")

# How mature is this project? Common values are
#   3 - Alpha
#   4 - Beta
#   5 - Production/Stable
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: Apache Software License
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Information Analysis
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

EXCLUDE_FROM_PACKAGES = ['']


def setup_package():
    write_version_py(filename=VERSION_PY)

    metadata = dict(
        name='bert_embedding',
        url='https://github.com/imgarylai/bert_embedding',
        download_url='https://github.com/imgarylai/bert_embedding/tree/master',
        author='Gary Lai',
        author_email='gary@gary-lai.com',
        description='BERT token level embedding with MxNet',
        long_description=open('README.md', 'r').read(),
        long_description_content_type='text/markdown',
        keywords='bert nlp mxnet gluonnlp machine deep learning sentence encoding embedding',
        license='ALv2',
        packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
        package_data={'': ['*.txt', '*.json']},
        install_requires=[
            'typing==3.6.6',
            'numpy==1.14.6',
            'mxnet==1.4.0',
            'gluonnlp==0.6.0',
        ],
        extras_require={
            'gpu': ['mxnet-cu92==1.3.0'],
        },
        tests_require=[
            'pytest',
        ],
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        scripts=['bin/bert'],
    )
    metadata['version'] = get_version_info(filename=VERSION_PY)[0]

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
