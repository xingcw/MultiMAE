import setuptools

__version__ = '0.0.1'

pkgs = setuptools.find_packages()

pkg_dirs = {}
new_pkgs = []

for pkg in pkgs:
    
    pkg_dir = "./" + "/".join(pkg.split("."))
    
    if pkg != "multimae":
        pkg = "multimae." + pkg
        
    new_pkgs.append(pkg)
    pkg_dirs.update({pkg: pkg_dir})

print(pkgs)
print(pkg_dirs)

setuptools.setup(
    name='multimae',
    version=__version__,
    packages=new_pkgs,
    package_dir=pkg_dirs,
)