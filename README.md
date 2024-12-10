## FEEL THE JOLLY SPIRIT YOU MORTAL!!!

This is a project made in rust using [winit](https://github.com/rust-windowing/winit) and [softbuffer](https://github.com/rust-windowing/softbuffer). The entirety of rendering code is written from scratch (unfortunately that also means it renders on the CPU, but at this scale rendering is not a bottleneck)
## Demo

You can find the demo of the web version here: https://mhanak.net/christmas

## Run it yourself
#### Native
To run the native desktop version run
``` bash
cargo run --release
```
#### Web Assembly
Feeling fancy? why not put that tree on the interwebs? To compile it to web assembly run:
``` bash
wasm-pack build --target web
```
this will create the `pkg` folder, where your `.wasm` is stored. you can use it together with `index.html` and `index.css` throught a web server. for example:
``` bash
python3 -m http.server
```

## How does it work?

It operates on similar principles to how 3D graphics is normally rendered. I have vertices, of which positions i multiply by the approperate martices (in my case they are 3-dimensional not 4) and project them onto the screen. Then instead of triangles i draw lines and circles.
Other than that it is a glorified particle system

## Known Issues

The web version is kind of broken on IOS, because of the way Safari handles HTML Canvas scaling, and for some reason it doesn't trigger touch events. Nonetheless, it will still work, it will just be a blurry mess.