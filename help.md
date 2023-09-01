For Google Cloud ssh sessions to remain open:
```bash
sudo /sbin/sysctl -w net.ipv4.tcp_keepalive_time=60 net.ipv4.tcp_keepalive_intvl=60 net.ipv4.tcp_keepalive_probes=5
sudo echo -e 'net.ipv4.tcp_keepalive_time=60\nnet.ipv4.tcp_keepalive_intvl=60\nnet.ipv4.tcp_keepalive_probes=5' >> /etc/sysctl.conf
```