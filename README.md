# xcelerator

Description will be added soon.

### profile_tf

### explorer

### report

### mobilenet_v1

## How to install
### Dependencies

Install the following dependencies first in the given sequence (recommended)
* python-adb
    * libssl-dev (apt-get)
    * rsa (pip)
    * PycryptoDome (pip)
    * python-m2crypto (apt-get)
    * adb (pip)
* plotly (pip)
* tensorflow (pip)

### Check you device
* Connect your android device, and check if it is accessible
* Copy the benchmark, label and image in /data/local/tmp/ (only one time)
* Execute adb-kill server
* Check the correctness of program by executing python test.py (no arguments)

### Confirm the results
Lookout for csv and html files generated
