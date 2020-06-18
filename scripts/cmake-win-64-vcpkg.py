import json,os,subprocess
from os import path

cmakePath = "cmake"
cudaPath = ""
vcpkgDir = ""

if(path.exists("CMakeSettings.json")):
    with open("CMakeSettings.json","r") as file:
        data = file.read()
    jsonData = json.loads(data)
    cmakePath=jsonData["cmake exe"]
    cudaPath=jsonData["cuda toolkit"]
    vcpkgDir=jsonData["vcpkg dir"]

command =[cmakePath, '-S',"../","-B","../"]

if(vcpkgDir!=""):
    command.append(
        "-DCMAKE_TOOLCHAIN_FILE={0}\\scripts\\buildsystems\\vcpkg.cmake".format(vcpkgDir)
    )

if(cudaPath):
    command.append(
        "-DCUDA_TOOLKIT_ROOT_DIR={0}".format(vcpkgDir)
    )

out = subprocess.Popen(
            command,
           stdout=subprocess.PIPE)

while out.poll() is None:
    l = out.stdout.read() # This blocks until it receives a newline.
    print(l.decode("utf-8"),end="")

os.system('pause')