# underworld3

## Documentation

<!-- Needs to be updated -->

The underworld documentation is in two parts: the user manual / theory manual is a jupyterbook that is built from this repository automatically from the sources in the `Jupyterbook` directory. The api documentation is also autogenerated.

<!-- - https://underworldcode.github.io/underworld3/main/FrontPage.html -->
- https://underworldcode.github.io/underworld3/main_api/index.html

The development branch has similar documentation:

<!-- - https://underworldcode.github.io/underworld3/development/FrontPage.html -->
- https://underworldcode.github.io/underworld3/development_api/index.html

## Binder demonstration version

 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/underworldcode/underworld3/development)
 
 [Binder - Dev Branch](https://mybinder.org/v2/gh/underworldcode/underworld3/development)


## Building

Refer to the Dockerfile for uw3 build instructions.  

To install from the repository
```shell
pip install .
```

The in-place `pip` installation may be helpful for developers (after the above)


```shell
pip install -e .
```

To clean the git repository or all files ... be careful.
```shell
./clean.sh
```

## Testing

Run the script
```shell
./test.sh
```
