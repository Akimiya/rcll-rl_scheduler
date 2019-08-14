
import os
import importlib


# find all module files
modules = []
for f in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if f.endswith('.py') and not f.endswith("__.py"):
        modules.append(f)

def reformat_imports(files):
    current_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

    # convert all imports inside them to "from . import ..."
    candidates = ["import " + x.strip(".py") for x in files]
    for f_in in files:
        f_out = f_in + "_"
        with open(current_dir + f_in, "r") as module_in:
            with open(current_dir + f_out, "w") as module_out:
                # early stop
                found_lines = 0
                # loop original file
                for line in module_in:
                    found_lines_last = found_lines

                    for imp_str in candidates:
                        if line.startswith(imp_str):
                            line = "from . " + line
                            found_lines += 1
                            break

                    # close early
                    if found_lines_last != 0 and found_lines_last == found_lines:
                        # continue from current position and write
                        r = module_in.read()
                        module_out.write(r)
                        #print("closing early [" + line + "] ==> " + r)
                        break

                    module_out.write(line)
        os.rename(current_dir + f_out, current_dir + f_in)

reformat_imports(modules)