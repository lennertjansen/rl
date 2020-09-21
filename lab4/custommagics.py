import io
import os
import re
from IPython.core import magic_arguments
from IPython.core.magic import cell_magic, Magics, magics_class

@magics_class
class CustomMagics(Magics):
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '-a', '--append', action='store_true', default=False,
        help='Append contents of the cell to an existing file. '
             'The file will be created if it does not exist.'
    )
    @magic_arguments.argument(
        'filename', type=str,
        help='file to write'
    )
    @cell_magic
    def execwritefile(self, line, cell):
            """Write the contents of the cell to a file.

            The file will be overwritten unless the -a (--append) flag is specified.
            """
            args = magic_arguments.parse_argstring(self.execwritefile, line)
            if re.match(r'^(\'.*\')|(".*")$', args.filename):
                filename = os.path.expanduser(args.filename[1:-1])
            else:
                filename = os.path.expanduser(args.filename)

            if os.path.exists(filename):
                if args.append:
                    print("Appending to %s" % filename)
                else:
                    print("Overwriting %s" % filename)
            else:
                print("Writing %s" % filename)

            mode = 'a' if args.append else 'w'
            with io.open(filename, mode, encoding='utf-8') as f:
                f.write(cell)
            get_ipython().run_cell(cell)
