# Tutorial to set up program on Raspberry Pi 5

## Python program

* 1.) Connect your device with the Raspberry Pi
Make sure your device is connected to the same network as the Raspberry Pi. You can connect your device on the PowerShell with:

```
ssh technorama@technorama.local
```

* 2.) If you haven't done that yet, clone this repository to your Raspberry Pi. You can find the instructions for that in the [Git-Instructions File](./Git-Instructions.md)

* 3.) To set up the connection to the docking station you'll need to use following command:
```
bash facial_processing/Config/display_setup.sh
```

* 4.) To set everything else up you'll need to use this command:
```
bash facial_processing/Config/setup.sh
```
*it might take some time to set everything up. The Pi will reboot automatically at the end of the script*

---