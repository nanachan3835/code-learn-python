import subprocess
import time
import os
import signal


def execSubProc(args):
    start_time = time.time()
    subproc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subproc.wait()
    end_time = time.time()
    execTime = end_time-start_time
    stdo = [str(s, 'UTF-8').strip() for s in subproc.stdout.readlines()]
    stdo = "\n".join(stdo)
    stde = [str(s, 'UTF-8').strip() for s in subproc.stderr.readlines()]
    stde = "\n".join(stde)
    print(f"Command        : {args}\n")
    print(f"Output         : {len(stdo)} bytes\n{stdo}\n")
    print(f"Error          : {len(stde)} bytes\n{stde}\n")
    print(f"Execution time : {execTime} seconds\n")
    print("--------------------------------------------------------------------")


pyfiles = ["test-cases-generator.py", "solo-mapping.py",
           "simple-combo-mapping.py", "maxima-combo-mapping.py"]

print("SFC-PHY Mapping Problem | Simulation")
print("--------------------------------------------------------------------")
for f in pyfiles:
    print(f">>> {f}")
    execSubProc(args=["python", f])