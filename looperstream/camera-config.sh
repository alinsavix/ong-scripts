#!/bin/bash
v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c gain=81
v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c brightness=128
v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c contrast=32
v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c saturation=101
v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c sharpness=24

v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c white_balance_automatic=0
v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c white_balance_temperature=5400

v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c power_line_frequency=1  # 50Hz
v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c backlight_compensation=0

v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c auto_exposure=1   # manual
v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c exposure_time_absolute=536    # 10ths of ms
v4l2-ctl -d /dev/v4l/by-id/usb-046d_0825_8B322E20-video-index0 -c exposure_dynamic_framerate=0
