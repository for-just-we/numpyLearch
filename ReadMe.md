
This project is to covert the model in [learch](https://github.com/eth-sri/learch) into numpy format because we need to implement it based on [NumCPP](https://github.com/dpilger26/NumCpp).

NumCPP is dependent on [boost](https://www.boost.org/) library. However, you don't need to link any libraries when using the 2 libs. Since they are header-based libs. You only need to set `NUMCPP_INCLUDE_DIR` and `BOOST_INCLUDE_DIR` when using cmake.

- `BOOST_INCLUDE_DIR` is set to the root dir of boost project.

- `NUMCPP_INCLUDE_DIR` is set to `<NUMCPP_ROOT>/include`, where `<NUMCPP_ROOT>` is the root dir of numCpp project.

