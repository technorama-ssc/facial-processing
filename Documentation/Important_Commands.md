# Important Commands

## Display 

Screen size and positions:
````commandline
DISPLAY=:0 xrandr
````

## VCGEN

Temperature:
````commandline
vcgencmd measure_temp
````

## Systemctl commands

Last 50 logs:
````commandline
journalctl -u facial_processing.service -n 50 -f
journalctl -u facial_processing.service -f
````


## Camera:

Camera info:
````commandline
rpicam-still --list-cameras
````

Kill stuck camera:
````commandline
sudo pkill -f rpicam-vid
sudo pkill -f ffmpeg
````





