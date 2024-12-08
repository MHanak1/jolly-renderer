// ----------------------------------------------------------------------------
// When compiling for web:
#[cfg(target_arch = "wasm32")]
use log::Level;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{self, prelude::*};

#[cfg(target_arch = "wasm32")]
mod main;

/// This is the entry-point for all the web-assembly.
/// This is called once from the HTML.
/// It loads the app, installs some callbacks, then returns.
/// You can add more callbacks like this if you want to call in to your code.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    use std::panic;
    // Here we ensure console_log is working
    console_log::init_with_level(Level::Debug).expect("could not initialize logger");
    // Here we ensure panic will send log to the web console
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    // Here we run our async application
    main::main();

    Ok(())
}
