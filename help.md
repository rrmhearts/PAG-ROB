May need to comment out the following
```
~/.local/lib/python3.10/site-packages/captum$ grep -Rnw . -e "#.*grid"
./attr/_utils/visualization.py:250:    # plt_axis.grid(b=False)
```

For Google Cloud ssh sessions to remain open:
```bash
sudo /sbin/sysctl -w net.ipv4.tcp_keepalive_time=60 net.ipv4.tcp_keepalive_intvl=60 net.ipv4.tcp_keepalive_probes=5
sudo echo -e 'net.ipv4.tcp_keepalive_time=60\nnet.ipv4.tcp_keepalive_intvl=60\nnet.ipv4.tcp_keepalive_probes=5' >> /etc/sysctl.conf
```
