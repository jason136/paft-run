use i2cdev::core::I2CDevice;
use i2cdev::linux::LinuxI2CDevice;

use crate::Error;

const ATTINY_I2C_ADDR: u16 = 0x08;

const LEFT_CMD: u8 = 0x00;
const RIGHT_CMD: u8 = 0x01;

pub struct Actuators {
    dev: LinuxI2CDevice,
}

impl Actuators {
    pub fn new() -> Result<Self, Error> {
        let dev = LinuxI2CDevice::new("/dev/i2c-1", ATTINY_I2C_ADDR)?;
        tracing::info!("ATtiny HAT I²C at 0x{:02X}", ATTINY_I2C_ADDR);
        Ok(Actuators { dev })
    }

    pub fn encode_wire(speed: i32) -> u8 {
        let s = speed.clamp(-90, 90);
        (s.saturating_add(90)).clamp(0, 180) as u8
    }

    pub fn actuate(&mut self, left_speed: i32, right_speed: i32) -> Result<(), Error> {
        self.set_left_speed(left_speed)?;
        self.set_right_speed(right_speed)?;
        Ok(())
    }

    pub fn set_left_speed(&mut self, speed: i32) -> Result<(), Error> {
        self.write_speed(LEFT_CMD, speed)
    }

    pub fn set_right_speed(&mut self, speed: i32) -> Result<(), Error> {
        self.write_speed(RIGHT_CMD, speed)
    }

    fn write_speed(&mut self, cmd: u8, speed: i32) -> Result<(), Error> {
        let b = Self::encode_wire(speed);
        self.dev.smbus_write_byte_data(cmd, b)?;
        tracing::trace!("servo cmd=0x{:02X} wire={} speed={}", cmd, b, speed);
        Ok(())
    }
}
