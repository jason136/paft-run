use std::time::Duration;

use i2cdev::core::I2CDevice;
use i2cdev::linux::LinuxI2CDevice;
use tokio::time::sleep;

use crate::Error;

const LSM6DSL_ADDRESS: u16 = 0x6A;

const LSM6DSL_CTRL1_XL: u8 = 0x10;
const LSM6DSL_CTRL2_G: u8 = 0x11;
const LSM6DSL_CTRL3_C: u8 = 0x12;

const LSM6DSL_OUTX_L_G: u8 = 0x22;
const LSM6DSL_OUTX_L_XL: u8 = 0x28;

pub async fn i2c_imu() -> Result<(), Error> {
    let mut dev = LinuxI2CDevice::new("/dev/i2c-1", LSM6DSL_ADDRESS)?;

    dev.smbus_write_byte_data(LSM6DSL_CTRL3_C, 0x01)?;
    sleep(Duration::from_millis(100)).await;

    dev.smbus_write_byte_data(LSM6DSL_CTRL1_XL, 0b01000000)?;

    dev.smbus_write_byte_data(LSM6DSL_CTRL2_G, 0b01000000)?;

    sleep(Duration::from_millis(100)).await;

    println!("LSM6DSL initialized successfully");

    loop {
        let acc_raw = dev.smbus_read_i2c_block_data(LSM6DSL_OUTX_L_XL, 6)?;

        let acc_x = i16::from_le_bytes([acc_raw[0], acc_raw[1]]);
        let acc_y = i16::from_le_bytes([acc_raw[2], acc_raw[3]]);
        let acc_z = i16::from_le_bytes([acc_raw[4], acc_raw[5]]);

        let gyr_raw = dev.smbus_read_i2c_block_data(LSM6DSL_OUTX_L_G, 6)?;

        let gyr_x = i16::from_le_bytes([gyr_raw[0], gyr_raw[1]]);
        let gyr_y = i16::from_le_bytes([gyr_raw[2], gyr_raw[3]]);
        let gyr_z = i16::from_le_bytes([gyr_raw[4], gyr_raw[5]]);

        let acc_scale = 0.000061; // Convert to g
        let acc = [
            acc_x as f32 * acc_scale,
            acc_y as f32 * acc_scale,
            acc_z as f32 * acc_scale,
        ];

        let gyr_scale = 0.00875; // Convert to dps
        let gyr = [
            gyr_x as f32 * gyr_scale,
            gyr_y as f32 * gyr_scale,
            gyr_z as f32 * gyr_scale,
        ];

        println!(
            "Acc (g): [{:.3}, {:.3}, {:.3}] | Gyr (dps): [{:.3}, {:.3}, {:.3}]",
            acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2]
        );

        sleep(Duration::from_millis(100)).await;
    }
}
