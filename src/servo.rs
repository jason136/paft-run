use gpio_cdev::Chip;

use crate::Error;

pub struct Servos {}

impl Servos {
    pub fn new() -> Result<Self, Error> {
        let chip = Chip::new("/dev/gpiochip0")?;

        Ok(Servos {})
    }

    pub fn actuate(&mut self, left_angle: f32, right_angle: f32) -> Result<(), Error> {
        Ok(())
    }
}
