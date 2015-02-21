"""Safe execution of untrusted Python code."""

import logging
import os.path
import shutil
import sys
import textwrap
import pickle
import json

from codejail import jail_code
from codejail.util import temp_directory, change_directory

log = logging.getLogger(__name__)


# Flags to let developers temporarily change some behavior in this file.

# Set this to True to log all the code and globals being executed.
LOG_ALL_CODE = False
# Set this to True to use the unsafe code, so that you can debug it.
ALWAYS_BE_UNSAFE = False

class SafeExecException(Exception):
    """
    Python code running in the sandbox has failed.

    The message will be the stdout of the sandboxed process, which will usually
    contain the original exception message.

    """
    pass


def safe_exec(code, globals_dict, files=None, python_path=None, slug=None,
              extra_files=None, settings_code=None, extra_imports=None):
    """
    Execute code as "exec" does, but safely.

    `code` is a string of Python code.  `globals_dict` is used as the globals
    during execution.  Modifications the code makes to `globals_dict` are
    reflected in the dictionary on return.

    `files` is a list of file paths, either files or directories.  They will be
    copied into the temp directory used for execution.  No attempt is made to
    determine whether the file is appropriate or safe to copy.  The caller must
    determine which files to provide to the code.

    `python_path` is a list of directory or file paths.  These names will be
    added to `sys.path` so that modules they contain can be imported.  Only
    directories and zip files are supported.  If the name is not provided in
    `extras_files`, it will be copied just as if it had been listed in `files`.

    `slug` is an arbitrary string, a description that's meaningful to the
    caller, that will be used in log messages.

    `extra_files` is a list of pairs, each pair is a filename and a bytestring
    of contents to write into that file.  These files will be created in the
    temp directory and cleaned up automatically.  No subdirectories are
    supported in the filename.

    Returns None.  Changes made by `code` are visible in `globals_dict`.  If
    the code raises an exception, this function will raise `SafeExecException`
    with the stderr of the sandbox process, which usually includes the original
    exception message and traceback.

    """
    the_code = []

    files = list(files or ())
    extra_files = extra_files or ()
    python_path = python_path or ()

    extra_names = set(name for name, contents in extra_files)
    if isinstance(extra_imports, str) and len(extra_imports) > 0:
        the_code.append(textwrap.dedent(extra_imports).strip())

    if isinstance(settings_code, str) and len(settings_code) > 0:
        the_code.append(textwrap.dedent(settings_code).strip())

    the_code.append(textwrap.dedent(
        """
        import sys, numpy, pandas, json, math, pickle, random, traceback, io, matplotlib;matplotlib.use('svg')
        """
        # Read the code and the globals from the stdin.
        """
        data = sys.stdin.buffer.read()
        code, g_dict = pickle.loads(data)
        if type(g_dict) == str:
            g_dict = pickle.loads(g_dict)
        """))

    for pydir in python_path:
        pybase = os.path.basename(pydir)
        the_code.append("sys.path.append(%r)\n" % pybase)
        if pybase not in extra_names:
            files.append(pydir)

    the_code.append(textwrap.dedent(
        # Execute the sandboxed code.
        """
        try:
            import matplotlib.pyplot as plt;plt.clf();random.seed(0)
            exec(code, g_dict)
        except Exception as err:
            try:
                print(traceback.format_exc(0).encode("ascii", "ignore"))
            except UnicodeEncodeError:
                print("Unknown Error")
            raise err
        """
        # Clean the globals for sending back as JSON over stdout.
        """
        answer_key = "{0}{1}".format(SANDBOX_CORRECT_PREFIX,"correct_context")
        bad_keys = ("__builtins__", SANDBOX_CHECK_VARS_NAME, answer_key)
        def pickleable(v):
            try:
                pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                return False
            return True
        if ANSWER_MODE:
            p_dict = {
                k:v
                for k,v in g_dict.items()
                if pickleable(v) and k in g_dict[SANDBOX_CHECK_VARS_NAME]
            }
        else:
            p_dict = {
                k:v
                for k,v in g_dict.items()
                if pickleable(v) and k not in bad_keys and not k.startswith(SANDBOX_CORRECT_PREFIX)
            }

        def jsonable(v):
            try:
                val = json.dumps(v)
            except Exception:
                return False
            if len(val) > 100:
                return False
            return True

        def stringable(v):
            try:
                m = str(v)
            except Exception:
                return False
            if len(m) > 0 and len(m) < 100:
                return True
            return False

        def typeable(v):
            try:
                m = str(type(v))
            except Exception:
                return False
            return True

        def jsonable_nolength(v):
            try:
                val = json.dumps(v)
            except Exception:
                return False
            return True

        def stringable_nolength(v):
            try:
                m = str(v)
            except Exception:
                return False
            return True

        v_dict = {}
        for k, v in g_dict.items():
            if k in bad_keys or k.startswith(SANDBOX_CORRECT_PREFIX):
                continue
            elif k.startswith(DATAQUEST_PLOT_PREFIX):
                v_dict[k] = v.getvalue()
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                v_dict[k] = "None"
            elif jsonable(v):
                v_dict[k] = v
            elif stringable(v):
                v_dict[k] = str(v)
            elif typeable(v):
                v_dict[k] = str(type(v))

        def make_real_var(k, v, bad_keys):
            if k in bad_keys or k.startswith(SANDBOX_CORRECT_PREFIX):
                return None
            elif isinstance(v, io.StringIO):
                return v.getvalue()
            elif jsonable_nolength(v):
                return v
            elif stringable_nolength(v):
                return str(v)
            elif typeable(v):
                return str(type(v))
            return None

        real_vars = {}
        for k, v in g_dict.items():
            val = make_real_var(k, v, bad_keys)
            if val is not None:
                real_vars[k] = val

        plots = []
        for var in g_dict[SANDBOX_CHECK_VARS_NAME]:
            if var.startswith(DATAQUEST_PLOT_PREFIX):
                given_var = g_dict.get(var)
                if given_var is not None:
                    plots.append(given_var.getvalue().strip())

        incorrect_vars = {}
        correct_vars = {}
        if answer_key in g_dict:
            answer_g_dict = g_dict[answer_key]
            if type(answer_g_dict) == str:
                try:
                    answer_g_dict = pickle.loads(answer_g_dict)
                except Exception:
                    answer_g_dict = pickle.loads(eval(answer_g_dict))
            for var in g_dict[SANDBOX_CHECK_VARS_NAME]:
                given_var = g_dict.get(var)
                correct_var = answer_g_dict.get(var)
                val = make_real_var(var, correct_var, [])
                if val is not None:
                    correct_vars[var] = val
                variable_okay = given_var is not None and type(given_var) == type(correct_var)
                if isinstance(given_var, io.StringIO) and isinstance(correct_var, str):
                    variable_okay = True
                if variable_okay:
                    if isinstance(correct_var, numpy.ndarray):
                        equal = numpy.array_equal(given_var, correct_var)
                    elif isinstance(correct_var, pandas.DataFrame) or isinstance(correct_var, pandas.Series):
                        equal = given_var.equals(correct_var)
                    elif var.startswith(DATAQUEST_PLOT_PREFIX):
                        if isinstance(correct_var, io.StringIO):
                            new_val = correct_var.getvalue().strip()
                        else:
                            new_val = correct_var.strip()
                        equal = False
                        for p in plots:
                            similarity = 0
                            if len(p) != len(new_val):
                                continue
                            try:
                                for i, s in enumerate(p):
                                    if s == new_val[i]:
                                        similarity += 1
                                if similarity/len(new_val) > .7:
                                    equal = True
                            except Exception:
                                pass
                    elif isinstance(correct_var, float):
                        # .sum() methods on numpy arrays vs the builtin sum function, among others, can have slight rounding differences.
                        # Adding this tolerance helps ensure those don't get flagged as incorrect.
                        equal = (correct_var - .001) <= given_var <= (correct_var + .001)
                    else:
                        equal = given_var == correct_var
                    variable_okay = variable_okay and equal
                if not variable_okay:
                    incorrect_vars[var] = {
                        "given_type": str(type(given_var)),
                        "correct_type": str(type(correct_var))
                    }
        """
        # Write the globals back to the calling process.
        """
        print("PICKLE_DATA:")
        d = pickle.dumps(p_dict, protocol=2)
        print(d)
        print("PICKLE_DATA:")
        json.dump(v_dict, sys.__stdout__)
        print("PICKLE_DATA:")
        json.dump(real_vars, sys.__stdout__)
        print("PICKLE_DATA:")
        json.dump(incorrect_vars, sys.__stdout__)
        print("PICKLE_DATA:")
        json.dump(correct_vars, sys.__stdout__)
        """
    ))

    stdin = pickle.dumps([code, globals_dict], protocol=pickle.HIGHEST_PROTOCOL)
    jailed_code = "".join(the_code)

    # Turn this on to see what's being executed.
    if LOG_ALL_CODE:        # pragma: no cover
        log.debug("Jailed code: %s", jailed_code)
        log.debug("Exec: %s", code)
        log.debug("Stdin: %s", stdin)

    res = jail_code.jail_code(
        "python", code=jailed_code, stdin=stdin, files=files, slug=slug,
        extra_files=extra_files,
    )
    if res.status != 0:
        log.error("Couldn't execute jailed code: %s" % res.stderr)
        output = res.stdout.decode("utf-8")
        if output is not None:
            output = output[:100000]
        display_vars = {}
        real_vars = {}
        incorrect_vars = {}
        correct_vars = {}
        data = ""
        error = True
    else:
        output = res.stdout.decode("utf-8")
        output, data, display_vars, real_vars, incorrect_vars, correct_vars = output.split("PICKLE_DATA:\n")
        display_vars = json.loads(display_vars)
        real_vars = json.loads(real_vars)
        incorrect_vars = json.loads(incorrect_vars)
        correct_vars = json.loads(correct_vars)
        error = False
    globals_dict = {"output": output, "data": data, "display_vars": display_vars, "real_vars": real_vars, "incorrect_vars": incorrect_vars, "error": error, "correct_vars": correct_vars}
    return globals_dict


def json_safe(d):
    """
    Return only the JSON-safe part of d.

    Used to emulate reading data through a serialization straw.

    """
    ok_types = (type(None), int, float, str, list, tuple, dict)
    bad_keys = ("__builtins__",)
    jd = {}
    for k, v in d.iteritems():
        if not isinstance(v, ok_types):
            continue
        if k in bad_keys:
            continue
        try:
            # Python's JSON encoder will produce output that
            # the JSON decoder cannot parse if the input string
            # contains unicode "unpaired surrogates" (only on Linux)
            # To test for this, we try decoding the output and check
            # for a ValueError
            json.loads(json.dumps(v))

            # Also ensure that the keys encode/decode correctly
            json.loads(json.dumps(k))
        except (TypeError, ValueError):
            continue
        else:
            jd[k] = v
    return json.loads(json.dumps(jd))


def not_safe_exec(code, globals_dict, files=None, python_path=None, slug=None,
                  extra_files=None):
    """
    Another implementation of `safe_exec`, but not safe.

    This can be swapped in for debugging problems in sandboxed Python code.

    This is not thread-safe, due to temporarily changing the current directory
    and modifying sys.path.

    """
    g_dict = json_safe(globals_dict)

    with temp_directory() as tmpdir:
        with change_directory(tmpdir):
            # Copy the files here.
            for filename in files or ():
                dest = os.path.join(tmpdir, os.path.basename(filename))
                shutil.copyfile(filename, dest)
            for filename, contents in extra_files or ():
                dest = os.path.join(tmpdir, filename)
                with open(dest, "w") as f:
                    f.write(contents)

            original_path = sys.path
            if python_path:
                sys.path.extend(python_path)
            try:
                exec(code, g_dict)
            except Exception as e:
                # Wrap the exception in a SafeExecException, but we don't
                # try here to include the traceback, since this is just a
                # substitute implementation.
                msg = "{0.__class__.__name__}: {0!s}".format(e)
                raise SafeExecException(msg)
            finally:
                sys.path = original_path


# If the developer wants us to be unsafe (ALWAYS_BE_UNSAFE), or if there isn't
# a configured jail for Python, then we'll be UNSAFE.
UNSAFE = ALWAYS_BE_UNSAFE or not jail_code.is_configured("python")

if UNSAFE:   # pragma: no cover
    # Make safe_exec actually call not_safe_exec, but log that we're doing so.

    def safe_exec(*args, **kwargs):                 # pylint: disable=E0102
        """An actually-unsafe safe_exec, that warns it's being used."""

        # Because it would be bad if this function were used in production,
        # let's log a warning when it is used.  Developers can live with
        # one more log line.
        slug = kwargs.get('slug', None)
        log.warning("Using codejail/safe_exec.py:not_safe_exec for %s", slug)

        return not_safe_exec(*args, **kwargs)
