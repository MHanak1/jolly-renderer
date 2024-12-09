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

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen]
    extern "C" {
        // Use `js_namespace` here to bind `console.log(..)` instead of just
        // `log(..)`
        #[wasm_bindgen(js_namespace = console)]
        fn log(s: &str);

        #[wasm_bindgen(js_namespace = console)]
        fn error(s: &str);

    }

    #[cfg(target_arch = "wasm32")]
    macro_rules! console_log {
        // Note that this is using the `log` function imported above during
        // `bare_bones`
        ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
    }

    #[cfg(target_arch = "wasm32")]
    macro_rules! console_error {
        // Note that this is using the `log` function imported above during
        // `bare_bones`
        ($($t:tt)*) => (error(&format_args!($($t)*).to_string()))
    }
    console_error!("start");

    // Here we ensure console_log is working
    //console_log::init_with_level(Level::Debug).expect("could not initialize logger");
    //wasm_logger::init(wasm_logger::Config::default().module_prefix("xmas-tree::main"));
    // Here we ensure panic will send log to the web console
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    // Here we run our async application
    main::run().await;

    Ok(())
}
