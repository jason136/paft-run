use std::os::unix::io::AsRawFd;
use std::time::Duration;

use i2cdev::core::I2CDevice;
use i2cdev::linux::{LinuxI2CDevice, LinuxI2CError};
use nix::libc;
use tokio::time::sleep;

use crate::Error;

const I2C_SLAVE: u64 = 0x0703;
const LSM9DS1_ADDRESS: u16 = 0x6A;

const LSM9DS1_CTRL_REG4: u8 = 0x1E;
const LSM9DS1_CTRL_REG1_G: u8 = 0x10;
const LSM9DS1_ORIENT_CFG_G: u8 = 0x13;

const LSM9DS1_CTRL_REG5_XL: u8 = 0x1F;
const LSM9DS1_CTRL_REG6_XL: u8 = 0x20;

const LSM6DSL_OUTX_L_XL: u8 = 0x28;
const LSM6DSL_OUTX_L_G: u8 = 0x22;

pub async fn i2c_imu() -> Result<(), Error> {
    let mut dev = LinuxI2CDevice::new("/dev/i2c-1", I2C_SLAVE)?;

    let result = unsafe { libc::ioctl(dev.as_raw_fd(), I2C_SLAVE, LSM9DS1_ADDRESS as libc::c_int) };

    if result < 0 {
        return Err(Error::from(std::io::Error::last_os_error()));
    }

    // Enable the accelerometer
    dev.smbus_write_byte_data(LSM9DS1_CTRL_REG5_XL, 0b00111000)?;
    dev.smbus_write_byte_data(LSM9DS1_CTRL_REG6_XL, 0b00101000)?;

    // Enable the gyroscope
    dev.smbus_write_byte_data(LSM9DS1_CTRL_REG4, 0b00111000)?;
    dev.smbus_write_byte_data(LSM9DS1_CTRL_REG1_G, 0b10111000)?;
    dev.smbus_write_byte_data(LSM9DS1_ORIENT_CFG_G, 0b10111000)?;

    sleep(Duration::from_secs(1));

    loop {
        let acc_raw = dev.smbus_read_i2c_block_data(LSM6DSL_OUTX_L_XL, 6)?;

        let acc_data = [
            (acc_raw[1] as i16) << 8 | acc_raw[0] as i16,
            (acc_raw[3] as i16) << 8 | acc_raw[2] as i16,
            (acc_raw[5] as i16) << 8 | acc_raw[4] as i16,
        ];

        let gyr_raw = dev.smbus_read_i2c_block_data(LSM6DSL_OUTX_L_G, 6)?;

        let gyr_data = [
            (gyr_raw[1] as i16) << 8 | gyr_raw[0] as i16,
            (gyr_raw[3] as i16) << 8 | gyr_raw[2] as i16,
            (gyr_raw[5] as i16) << 8 | gyr_raw[4] as i16,
        ];

        println!("Acc: {:?}, Gyr: {:?}", acc_data, gyr_data);

        sleep(Duration::from_millis(100)).await;
    }

    Ok(())
}
