# Google Compute HELP Page

For Google Cloud ssh sessions to remain open:

```bash
sudo /sbin/sysctl -w net.ipv4.tcp_keepalive_time=60 net.ipv4.tcp_keepalive_intvl=60 net.ipv4.tcp_keepalive_probes=5
sudo echo -e 'net.ipv4.tcp_keepalive_time=60\nnet.ipv4.tcp_keepalive_intvl=60\nnet.ipv4.tcp_keepalive_probes=5' >> /etc/sysctl.conf
```

## Using Google Cloud

Installation instructions for CUDA for Google Cloud: https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#os_support.

### Not recommended to use Debian

If you are using Debian...
Install cuda from the NVIDIA documentation: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_network

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
```
