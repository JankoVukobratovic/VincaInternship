"""Shared infrastructure of the neurips_submission package.

Import style used across the package (works both standalone and as a
package, provided the neurips_submission root is on sys.path - every
experiment entry point bootstraps that):

    from common import core, perturb, training, restore, classical, io_utils
"""
