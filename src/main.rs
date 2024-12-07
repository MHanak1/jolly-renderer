#![windows_subsystem = "windows"]

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::ops::{Add, Mul};
use std::rc::Rc;
use std::thread;
use winit::event::{ElementState, Event, MouseScrollDelta, StartCause, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Fullscreen, Window, WindowBuilder};
use dyn_clone::DynClone;
use image::{GenericImage, Pixel, RgbaImage};
use rand::random;
use itertools::Itertools;
use rand::seq::{SliceRandom};
use winit::keyboard::{Key, NamedKey};

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::__rt::Start;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::{HtmlCanvasElement};
#[cfg(target_arch = "wasm32")]
use winit::dpi::PhysicalSize;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::{EventLoopWindowTargetExtWebSys, PollStrategy};
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowBuilderExtWebSys;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;

#[path = "utils/winit_app.rs"]
mod winit_app;

#[derive(Clone)]
struct Matrix3<T: Mul + Clone> {
    mat: [[T; 3]; 3],
}

impl<T: Mul + Clone> Matrix3<T> {
    fn new(mat: [[T; 3]; 3]) -> Self {
        Self { mat }
    }
    fn get(&self, row: usize, col: usize) -> T {
        self.mat[row][col].clone()
    }
    fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.mat[row][col]
    }
}

impl Mul<f32> for Matrix3<f32> {
    type Output = Matrix3<f32>;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut new = self.clone();
        for i in 0..3 {
            for j in 0..3 {
                new.mat[i][j] = rhs * self.mat[i][j];
            }
        }
        new
    }
}

//shit's fucked but refactoring it would be a nightmare. matrix should be nice though
trait Position: DynClone {
    fn new(from: [f32; 3]) -> Self where Self: Sized;
    fn default() -> Self where Self: Sized;
    fn x(&self) -> f32;
    fn y(&self) -> f32;
    fn z(&self) -> f32;
    fn x_mut(&mut self) -> &mut f32;
    fn y_mut(&mut self) -> &mut f32;
    fn z_mut(&mut self) -> &mut f32;
    fn set_x(&mut self, x: f32);
    fn set_y(&mut self, y: f32);
    fn set_z(&mut self, z: f32);
    fn get(&self) -> [f32; 3];
    //fn scale(&self, scalar: f32) -> Self where Self: Sized;
    //fn mul(&self, matrix: [[f32; 3]; 3]) -> Self where Self: Sized;
    //fn offset(&self, offset: [f32; 3]) -> Self where Self: Sized;
}
dyn_clone::clone_trait_object!(Position);

trait Transformable {
    fn scale(&self, scalar: f32) -> Vector3;
    fn mul(&self, matrix: Matrix3<f32>) -> Vector3;
    fn offset(&self, offset: [f32; 3]) -> Vector3;
}
#[derive(Clone)]
struct ScreenPosition {
    pos: [f32; 2],
    z: f32
}

impl Position for ScreenPosition {
    fn new(from: [f32; 3]) -> Self where Self: Sized {
        Self { pos: [from[0], from[1]], z: 0.0 }
    }

    fn default() -> Self { Self { pos: [0.0, 0.0], z: 0.0 } }

    fn x(&self) -> f32 {
        self.pos[0]
    }

    fn y(&self) -> f32 {
        self.pos[1]
    }

    fn z(&self) -> f32 {
        self.z
    }

    fn x_mut(&mut self) -> &mut f32 {
        &mut self.pos[0]
    }
    fn y_mut(&mut self) -> &mut f32 {
        &mut self.pos[1]
    }
    fn z_mut(&mut self) -> &mut f32 {
        &mut self.z
    }

    fn set_x(&mut self, x: f32) {
        self.pos[0] = x;
    }

    fn set_y(&mut self, y: f32) {
        self.pos[1] = y;
    }
    fn set_z(&mut self, z: f32) {
        self.z = z;
    }

    fn get(&self) -> [f32; 3] {
        [self.x(), self.y(), self.z()]
    }
}



#[derive(Clone)]
struct Vector3 {
    pos: [f32; 3],
}

impl Position for Vector3 {
    fn new(from: [f32; 3]) -> Self
    where
        Self: Sized
    {
        Self { pos: from }
    }

    fn default() -> Self
    where
        Self: Sized
    { Self { pos: [0.0, 0.0, 0.0] } }

    fn x(&self) -> f32 {
        self.pos[0]
    }

    fn y(&self) -> f32 {
        self.pos[1]
    }

    fn z(&self) -> f32 {
        self.pos[2]
    }

    fn x_mut(&mut self) -> &mut f32 {
        &mut self.pos[0]
    }
    fn y_mut(&mut self) -> &mut f32 {
        &mut self.pos[1]
    }
    fn z_mut(&mut self) -> &mut f32 {
        &mut self.pos[2]
    }

    fn set_x(&mut self, x: f32) {
        self.pos[0] = x;
    }

    fn set_y(&mut self, y: f32) {
        self.pos[1] = y;
    }
    fn set_z(&mut self, z: f32) {
        self.pos[2] = z;
    }

    fn get(&self) -> [f32; 3] {
        [self.x(), self.y(), self.z()]
    }
}
impl Transformable for Vector3 {
    fn scale(&self, scalar: f32) -> Vector3 {
        let mut new_pos = self.clone();
        for i in 0..3 {
            new_pos.pos[i] *= scalar;
        }
        new_pos
    }

    fn mul(&self, matrix: Matrix3<f32>) -> Vector3 {
        let mut new_pos = self.clone();
        for i in 0..3 {
            new_pos.pos[i] = self.pos[0] * matrix.get(i, 0) + self.pos[1] * matrix.get(i, 1) + self.pos[2] * matrix.get(i, 2);
        }
        new_pos
    }

    fn offset(&self, offset: [f32; 3]) -> Vector3 {
        let mut new_pos = self.clone();
        for (i, offset) in offset.iter().enumerate() {
            new_pos.pos[i] = self.pos[i] + offset;
        }
        new_pos
    }
}

trait TransformMatrix {
    fn get_rotation_matrix(&self) -> Matrix3<f32>;
}

struct XAxisRotation {
    angle: f32,
}

struct YAxisRotation {
    angle: f32,
}

struct ZAxisRotation {
    angle: f32,
}

struct ScaleTransform {
    scale: [f32; 3],
}

impl TransformMatrix for XAxisRotation {
    fn get_rotation_matrix(&self) -> Matrix3<f32> {
        let rad = self.angle / 180.0 * std::f32::consts::PI;
        Matrix3 {
            mat: [[1.0, 0.0, 0.0],
                [0.0, f32::cos(rad), -f32::sin(rad)],
                [0.0, f32::sin(rad), f32::cos(rad)]]
        }
    }
}

impl TransformMatrix for YAxisRotation {
    fn get_rotation_matrix(&self) -> Matrix3<f32> {
        let rad = self.angle / 180.0 * std::f32::consts::PI;
        Matrix3 {
            mat:[
                [f32::cos(rad), 0.0, f32::sin(rad)],
                [0.0, 1.0, 0.0],
                [-f32::sin(rad), 0.0, f32::cos(rad)]
            ]
        }
    }
}

impl TransformMatrix for ZAxisRotation {
    fn get_rotation_matrix(&self) -> Matrix3<f32> {
        let rad = self.angle / 180.0 * std::f32::consts::PI;
        Matrix3 {
            mat:[
                [f32::cos(rad), -f32::sin(rad), 0.0],
                [f32::sin(rad), f32::cos(rad), 0.0],
                [0.0, 0.0, 1.0]
            ]
        }
    }
}

impl TransformMatrix for ScaleTransform {
    fn get_rotation_matrix(&self) -> Matrix3<f32> {
        Matrix3 {
            mat: [
                [self.scale[0], 0.0, 0.0],
                [0.0, self.scale[1], 0.0],
                [0.0, 0.0, self.scale[2]]
            ]
        }
    }
}

impl XAxisRotation {
    fn from (angle: f32) -> Self {
        Self { angle }
    }
}

impl YAxisRotation {
    fn from (angle: f32) -> Self {
        Self { angle }
    }
}

impl ZAxisRotation {
    fn from (angle: f32) -> Self {
        Self { angle }
    }
}

impl ScaleTransform {
    fn from (scale: [f32; 3]) -> Self {
        Self { scale }
    }

    fn from_scalar (scale: f32) -> Self {
        Self { scale: [scale, scale, scale]}
    }
}



#[derive(Clone)]
struct Font {
    font_image: RgbaImage,
    width: u32,
    has_uppercase: bool,
}

impl Font {
    fn get_character(&self, character: char) -> Option<RgbaImage> {
        //TODO: clean up
        let mut char_index = character as i32 - 'A' as i32;
        if char_index < 0 || character as usize > 'z' as usize {
            return None;
        }
        if character > 'Z' {
            char_index -= 6;
        }
        if !self.has_uppercase {
            char_index %= 26;
        }
        Some(RgbaImage::from(
            self.font_image.clone().sub_image(
                char_index as u32 * self.width,
                0,
                self.width,
                self.font_image.height(),
            ).to_image())
        )
    }
}

trait DrawableElement: DynClone {
    fn update(&mut self, dt: Duration);
    fn render(&self, renderer: &Renderer, buffer: &mut [u32], depth_buffer: &mut [f32]);
    fn set_size(&mut self, size: f32);
    fn get_color(&self) -> [u8; 4];
    fn set_color(&mut self, color: [u8; 4]);
    fn transform_by(&mut self, matrix: Matrix3<f32>);
    fn scale_by(&mut self, scalar: f32);
    fn offset(&mut self, offset: [f32; 3]);
    fn get_pos(&self) -> Vector3;
    fn get_depth(&self) -> f32;
}
dyn_clone::clone_trait_object!(DrawableElement);

#[derive(Clone)]
struct Line {
    start: Vector3,
    end: Vector3,
    width: f32,
    color: [u8; 4],
    animations: Vec<Box<dyn ElementAnimation<Line>>>,
}

impl Line {
    fn new (start: Vector3, end: Vector3, color: [u8; 4], animations: Vec<Box<dyn ElementAnimation<Line>>>) -> Self {
        Line {
            start,
            end,
            width: 0.0,
            color,
            animations,
        }
    }
}

impl DrawableElement for Line {
    //TODO: unfuck this code (i mean it works but it really hurts to look at)
    fn update(&mut self, time_delta: Duration) {
        let mut new = self.clone();
        for animation in self.animations.iter_mut() {
            let new1 = animation.update(new.clone(), time_delta);
            new = new1;
        }
        *self = new;
    }

    fn render(&self, renderer: &Renderer, buffer: &mut [u32], depth_buffer: &mut [f32]) {
        let line_start = renderer.get_pixel_pos([self.start.x(), self.start.y()]);
        let line_end = renderer.get_pixel_pos([self.end.x(), self.end.y()]);
        renderer.draw_line(buffer, depth_buffer, (line_start[0], line_start[1], self.start.z()), (line_end[0], line_end[1], self.end.z()), self.color);
    }

    fn set_size(&mut self, size: f32) {
        self.width = size;
    }

    fn get_color(&self) -> [u8; 4] {
        self.color
    }

    fn set_color(&mut self, color: [u8; 4]) {
        self.color = color;
    }

    fn transform_by(&mut self, matrix: Matrix3<f32>) {
        self.start = self.start.mul(matrix.clone());
        self.end = self.end.mul(matrix.clone());
    }

    fn scale_by(&mut self, scalar: f32) {
        self.start = self.start.scale(scalar);
        self.end = self.end.scale(scalar);
    }

    fn offset(&mut self, offset: [f32; 3]) {
        self.start = self.start.offset(offset);
        self.end = self.end.offset(offset);
    }

    fn get_pos(&self) -> Vector3 {
        self.start.scale(0.5).offset(self.end.scale(0.5).pos) //this fuckery calculates the average (hopefully)
    }

    fn get_depth(&self) -> f32 {
        (self.start.z() + self.end.z()) / 2.0
    }
}

#[derive(Clone)]
struct TextElement {
    position: ScreenPosition,
    text: String,
    scale: f32,
    color: [u8; 4],
    font: Font
}

impl DrawableElement for TextElement {
    fn update(&mut self, _time_delta: Duration) {
        //do nothing
    }

    fn render(&self, renderer: &Renderer, buffer: &mut [u32], depth_buffer: &mut [f32]){
        let scale = i32::clamp(self.scale as i32, 1, i32::MAX);
        for character in 0..self.text.len() {
            let img = self.font.get_character(self.text.chars().nth(character).unwrap());
            if let Some(img) = img {
                for x in 0..img.width() {
                    for y in 0..img.height() {
                        let color = img.get_pixel(x, y).to_rgba();
                        /*
                        for i in 0..3 {
                            color.0[i] = (color.0[i] as f32 / 255.0 * self.color[i] as f32 / 255.0) as u8;
                            println!("{}: {}, {}", i, color.0[i] , self.color[i] );
                        }*/
                        for i in 0..scale{
                            for j in 0..scale {
                                let nx: i32 = x as i32 * scale
                                    + character as i32 * img.height() as i32 * scale
                                    + self.position.get()[0] as i32
                                    - self.text.len() as i32 * self.font.width as i32 / 2 * scale
                                    + renderer.width as i32 / 2
                                    + i;
                                let ny: i32 = y as i32 * scale
                                    + self.position.get()[1] as i32
                                    + j;

                                if self.get_depth() < renderer.get_depth_at(depth_buffer, [nx, ny]) {
                                    renderer.set_depth_at(depth_buffer, [nx, ny], self.get_depth());
                                    renderer.set_color_with_alpha(buffer, (nx, ny), color.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn set_size(&mut self, size: f32) {
        self.scale = size;
    }

    fn get_color(&self) -> [u8; 4] {
        self.color
    }

    fn set_color(&mut self, color: [u8; 4]) {
        self.color = color;
    }

    fn transform_by(&mut self, matrix: Matrix3<f32>) {
        //self.position = self.position.mul(matrix);
    }

    fn scale_by(&mut self, scalar: f32) {
        self.scale = self.scale.mul(scalar);
    }

    fn offset(&mut self, offset: [f32; 3]) {
        self.position.get()[0] += offset[0];
        self.position.get()[1] += offset[1];
        self.position.get()[2] += offset[2];
    }

    fn get_pos(&self) -> Vector3 {
        Vector3::new([self.position.x(), self.position.y(), self.position.z()])
    }

    fn get_depth(&self) -> f32 {
        self.position.z()
    }
}

#[derive(Clone)]
struct ParticleElement {
    position: Vector3,
    size: f32,
    color: [u8; 4],
    animations: Vec<Box<dyn ElementAnimation<ParticleElement>>>,
}

impl ParticleElement {
    fn new (position: Vector3, size: f32, color: [u8; 4], animations: Vec<Box<dyn ElementAnimation<Self>>>) -> ParticleElement {
        ParticleElement {
            position,
            size,
            color,
            animations,
        }
    }
}

impl DrawableElement for ParticleElement {
    fn update(&mut self, time_delta: Duration) {
        let mut new = self.clone();
        for animation in self.animations.iter_mut() {
            let new1 = animation.update(new.clone(), time_delta);
            new = new1;
        }
        *self = new;
    }

    fn render(&self, renderer: &Renderer, buffer: &mut [u32], depth_buffer: &mut [f32]){
        let pos = renderer.get_pixel_pos([self.position.x(), self.position.y()]);
        renderer.draw_circle(buffer, depth_buffer, (pos[0] as i32, pos[1] as i32, self.get_depth()), self.size * renderer.get_render_scale(), self.color)
    }

    fn set_size(&mut self, size: f32) {
        self.size = size;
    }

    fn get_color(&self) -> [u8; 4] {
        self.color
    }

    fn set_color(&mut self, color: [u8; 4]) {
        self.color = color;
    }

    fn transform_by(&mut self, matrix: Matrix3<f32>) {
        self.position = self.position.mul(matrix.clone());
    }

    fn scale_by(&mut self, scalar: f32) {
        self.position = self.position.scale(scalar);
    }

    fn offset(&mut self, offset: [f32; 3]) {
        self.position = self.position.offset(offset);
    }

    fn get_pos(&self) -> Vector3 {
        self.position.clone()
    }

    fn get_depth(&self) -> f32 {
        self.position.z()
    }
}
trait ElementAnimation<T: DrawableElement>: DynClone{
    fn update(&mut self, element: T, dt: Duration) -> T;
}
dyn_clone::clone_trait_object!(<T> ElementAnimation<T> where T: DrawableElement);

#[derive(Clone)]
struct RotateYAnimation {
    rate: f32,
}

impl RotateYAnimation {
    fn new(rate: f32) -> Self {
        Self { rate }
    }
}

impl<T: DrawableElement + DynClone + Clone> ElementAnimation<T> for RotateYAnimation {
    fn update(&mut self, element: T, dt: Duration) -> T {
        let mut new = element.clone();
        new.transform_by(YAxisRotation::from(dt.as_secs_f32() * self.rate).get_rotation_matrix());
        new
    }
}

#[derive(Clone)]
struct SnowAnimation {
    velocity: Vector3,
    fall_speed: f32,
    velocity_noise: f32,
    bounds: [Vector3; 2],
    opacity_falloff_distance: f32,
}

impl SnowAnimation {
    fn new(fall_speed: f32, velocity: Vector3, velocity_noise: f32, bounds: [Vector3; 2]) -> SnowAnimation {
        SnowAnimation {
            fall_speed,
            velocity,
            velocity_noise,
            bounds,
            opacity_falloff_distance: 1.0,
        }
    }
}

impl<T: DrawableElement + DynClone> ElementAnimation<T> for SnowAnimation {
    fn update(&mut self, element: T, dt: Duration) -> T {
        let mut bounds_dimensions = [0.0, 0.0, 0.0];

        for i in 0..3 {
            bounds_dimensions[i] = f32::abs(self.bounds[0].pos[i] - self.bounds[1].pos[i]);
        }
        let mut new: T = element;

        self.velocity.set_y(-self.fall_speed);
        if self.velocity_noise != 0.0 {
            self.velocity.set_x(f32::clamp(self.velocity.x() + (random::<f32>() - 0.5) * self.velocity_noise, -self.fall_speed, self.fall_speed));
            self.velocity.set_z(f32::clamp(self.velocity.z() + (random::<f32>() - 0.5) * self.velocity_noise, -self.fall_speed, self.fall_speed));
        }

        //println!("vel: [{}, {}, {}]", self.velocity.x(), self.velocity.y(), self.velocity.z());
        let mut distance_to_border = self.opacity_falloff_distance;

        new.offset(self.velocity.scale(dt.as_secs_f32()).pos);
        for i in 0..3{
            let (lbound, hbound) =
                (f32::min(self.bounds[0].pos[i], self.bounds[1].pos[i]),
                 f32::max(self.bounds[0].pos[i], self.bounds[1].pos[i]));

            distance_to_border = f32::min(distance_to_border, f32::abs(hbound - new.get_pos().pos[i]));
            distance_to_border = f32::min(distance_to_border, f32::abs(lbound - new.get_pos().pos[i]));


           if new.get_pos().pos[i] < lbound {
               let mut offset = [0.0, 0.0, 0.0];
               offset[i] = bounds_dimensions[i];
               new.offset(offset);
           }else if new.get_pos().pos[i] > hbound {
               let mut offset = [0.0, 0.0, 0.0];
               offset[i] = -bounds_dimensions[i];
               new.offset(offset);
           }
        }
        let mut new_color = new.get_color();
        new_color[3] = (distance_to_border / self.opacity_falloff_distance * 255.0) as u8;
        new.set_color(new_color);
        new
    }
}

struct DrawableObject {
    transforms: HashMap<u16, Box<dyn TransformMatrix>>,
    position: Box<dyn Position>,
    elements: Vec<Box<dyn DrawableElement>>,
}

impl DrawableObject {
    fn empty() -> DrawableObject {
        DrawableObject {
            transforms: HashMap::new(),
            position: Box::<Vector3>::new(Vector3::default()),
            elements: Vec::new()
        }
    }
    fn get_transformed_elements(&self) -> Vec<Box<dyn DrawableElement>>{
        let mut new_elements : Vec<Box<dyn DrawableElement>> = Vec::new();
        for element in self.elements.iter() {
            let mut new_element = element.clone();
            for matrix in self.transforms.values() {
                new_element.transform_by(matrix.get_rotation_matrix())
            }
            new_element.offset(self.position.get());
            new_elements.push(new_element);
        }
        new_elements
    }
}



struct Renderer {
    width: usize,
    height: usize,
    scale: f32,
    objects: HashMap<String, DrawableObject>,
    rotation: f32,
    position_offset: [f32; 2],
    transforms: HashMap<u16, Box<dyn TransformMatrix>>, //u16 is the id
}

impl Renderer{
    fn new(width: usize, height: usize) -> Renderer {
        Renderer {
            width,
            height,
            scale: 14.0,
            objects: HashMap::new(),
            rotation: 0.0,
            position_offset: [0.0; 2],
            transforms: HashMap::new(),
        }
    }

    fn render (&self, buffer: &mut [u32], depth_buffer: &mut [f32]) {
        buffer.fill(0);
        for container in self.objects.values() {
            for element in container.get_transformed_elements().iter() {
                let mut element = element.clone();
                for id in self.transforms.keys().sorted() {
                    element.transform_by(self.transforms[id].get_rotation_matrix())
                }
                element.render(self, buffer, depth_buffer);
            }
        }
    }

    fn update (&mut self, time_delta: Duration) {
        for container in self.objects.values_mut() {
            for element in container.elements.iter_mut() {
                element.update(time_delta);
            }
        }
    }

    fn get_render_scale (&self) -> f32 {
        self.scale * self.scale / 100.0 * self.height as f32 / 1080.0
    }

    fn get_depth_at(&self, depth_buffer: &[f32],  pos: [i32; 2]) -> f32 {
        if pos[0] > 0 && pos[0] < self.width as i32 && pos[1] > 0 && pos[1] < self.height as i32{
            return depth_buffer[(pos[0] + pos[1] * self.width as i32) as usize]
        }
        f32::INFINITY
    }

    fn set_depth_at(&self, depth_buffer: &mut [f32], pos: [i32; 2], value: f32) {
        if pos[0] > 0 && pos[0] < self.width as i32 && pos[1] > 0 && pos[1] < self.height as i32{
            //println!("x: {}, y: {}, width: {}, index: {}", pos[0], pos[1], self.width, (pos[0] + (pos[1] * self.width as i32)) as usize);
            depth_buffer[(pos[0] + pos[1] * self.width as i32) as usize] = value
        }
    }

    fn draw_line (&self, buffer: &mut [u32], depth_buffer: &mut [f32], (x_start, y_start, start_depth): (f32, f32, f32), (x_end, y_end, end_depth): (f32, f32, f32), color: [u8; 4]) {
//stolen from wikipedia
        let (mut x, mut y) = (x_start, y_start);
        let dx = f32::abs(f32::clamp((x_end - x) / (y_end - y), -1.0, 1.0)) * if x_start > x_end {-1.0} else {1.0};
        let dy = f32::abs(f32::clamp((y_end - y) / (x_end - x), -1.0, 1.0)) * if y_start > y_end {-1.0} else {1.0};

        let mut iters = 0;
        let mut progress = 0.0;

        while progress <= 1.0 {
            progress = f32::max(f32::abs(x - x_start), f32::abs(y - y_start)) / f32::max(f32::abs(y_start - y_end), f32::abs(x_start - x_end));
            //println!("progress: {}", progress);
            let depth = start_depth * (1.0 - progress) + end_depth * progress;
            if depth < self.get_depth_at(depth_buffer, [x as i32, y as i32]) {
                self.set_depth_at(depth_buffer, [x as i32, y as i32], depth);
                self.set_color_with_alpha(buffer, (x as i32, y as i32), color);
            }
            iters += 1;

            if iters > 10000 {
                println!("WARN: Too many line draw iterations. Aborting. (start: ({}, {}), end: ({}, {}), delta x: {}, delta y: {})", x_start, y_start, x_end, y_end, dx, dy);
                return
            }

            //x += if right {1.0} else {-1.0};
            x += dx;
            y += dy;
        }
    }

    fn draw_circle(&self, buffer: &mut [u32], depth_buffer: &mut [f32], (x, y, depth): (i32, i32, f32), radius: f32, color: [u8; 4]) {
        //check if the circle is on the screen
        if x as f32 + radius < 0.0 || y as f32 + radius < 0.0 || x as f32 - radius > self.width as f32  || y as f32 - radius > self.height as f32 {
            return;
        }
        for i in (-radius.floor() as i32)..(radius.ceil() as i32) {
            for j in (-radius.floor() as i32)..(radius.ceil() as i32) {
                //println!("i: {}, j: {}", i, j);
                if ((i * i + j * j) as f32) < radius * radius && depth < self.get_depth_at(depth_buffer, [x + i, y +j]) {
                    self.set_depth_at(depth_buffer, [x + i, y + j], depth);
                    self.set_color_with_alpha(buffer, (x + i, y + j), color);
                }
            }
        }
    }

    fn get_pixel_pos(&self, pos: [f32; 2]) -> [f32; 2] {
        [
            //TODO: clean this mess up
            (self.width / 2) as f32 + (pos[0] * self.get_render_scale() / self.width as f32 * 1080.0 * self.width as f32 / 10.0) + self.position_offset[0],
            (self.height / 2) as f32 + (pos[1] * self.get_render_scale() / self.height as f32 * 1080.0 * self.height as f32 / 10.0) + self.position_offset[1]
        ]
    }

    fn set_color_with_alpha (&self, buffer: &mut [u32], (x, y): (i32, i32), color: [u8; 4]) {
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 { return }
        let alpha = color[3] as f32 / 255.0;
        let mut new_color = self.get_color(buffer, (x as usize, y as usize));
        for i in 0..3 {
            new_color[i] = (new_color[i] as f32 * (1.0 - alpha)   +   color[i] as f32 * alpha) as u8
            //new_color[i] = f64::floor(new_color[i] as f64 * (1.0 - color[3] as f64) / 255.0 + color[i] as f64 * (color[3] as f64) / 255.0 ) as u8
        }
        self.set_color(buffer, (x, y), new_color)
    }

    fn set_color (&self, buffer: &mut [u32], (x, y): (i32, i32), color: [u8; 3]) {
        if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 { return }
        buffer[x as usize + y as usize * self.width] =
            ((color[0] as u32) << 16) +
            ((color[1] as u32) <<  8) +
            (color[2] as u32);
    }

    fn get_color (&self, buffer: &[u32], (x, y): (usize, usize)) -> [u8; 3]{
        let col: [u8; 4] = buffer[x + y * self.width].to_be_bytes();
        [col[1], col[2], col[3]]

    }
}


#[cfg(not(target_arch = "wasm32"))]
fn setup_window (event_loop: &EventLoop<()>) -> Rc<Window> {
    Rc::new(WindowBuilder::new().build(event_loop).unwrap())
    //window.set_min_inner_size(Some(LogicalSize::new(600.0, 400.0)));
}

#[cfg(target_arch = "wasm32")]
fn setup_window (event_loop: &EventLoop<()>) -> Rc<Window> {

    let window = web_sys::window().expect("no global `window` exists");
    let document = window.document().expect("should have a document on window");
    let body = document.body().expect("document should have a body");


    let canvas = document.create_element("canvas").unwrap();
    let canvas: HtmlCanvasElement = canvas
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| ())
        .unwrap();
    canvas.set_attribute("id", "canvas").unwrap();
    canvas.set_width(window.inner_width().unwrap().as_f64().unwrap() as u32);
    canvas.set_height(window.inner_height().unwrap().as_f64().unwrap() as u32);

    body.append_child(&canvas).unwrap();

    let context = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();
    context.scale(window.device_pixel_ratio(), window.device_pixel_ratio()).unwrap();

    let winit_window = Rc::new(
        WindowBuilder::new()
            .with_canvas(Some(canvas))
        .build(&event_loop).unwrap());
    // let _ = winit_window.request_inner_size(LogicalSize::new(800.0, 800.0));
    winit_window
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    #[cfg(target_arch = "wasm32")]
    event_loop.set_poll_strategy(PollStrategy::IdleCallback);

    let window = setup_window(&event_loop);

    let context = softbuffer::Context::new(window.clone()).unwrap();
    let mut surface = softbuffer::Surface::new(&context, window.clone()).unwrap();

    let mut renderer = Renderer::new(window.inner_size().width as usize, window.inner_size().height as usize);
    let mut last_render = Instant::now();
    let mut last_frame = Instant::now();
    //let mut depth_buffer = Vec::new();

    let font_bytes = include_bytes!("resources/font.png");

    let font = image::load_from_memory(font_bytes).unwrap().to_rgba8();

    renderer.objects.insert("tree".to_string(), DrawableObject::empty());
    renderer.objects.insert("text".to_string(), DrawableObject::empty());
    renderer.objects.insert("snow".to_string(), DrawableObject::empty());

    //TREE
    renderer.objects.get_mut("tree").unwrap().position = Box::<Vector3>::new(Vector3::new([0.0, -2.0, 0.0]));
    //renderer.containers.get_mut("tree").unwrap().position = Box::<Vector3>::new(Vector3::new([0.0, -1.2, 0.0]));

    let mut verts = [
        [0.0,	    0.0,	 1.0],
        [-0.866,    0.0,	 0.5],
        [-0.866,	0.0,	-0.5],
        [0.0,	    0.0,	-1.0],
        [0.866,	    0.0,	-0.5],
        [0.866,	    0.0,	 0.5],
        [0.0,	    0.8,	 0.4],
        [-0.346,    0.8,	 0.2],
        [-0.346,    0.8,	-0.2],
        [0.0,	    0.8,	-0.4],
        [0.346,	    0.8,	-0.2],
        [0.346,	    0.8,	 0.2],
    ];

    let line_color = [0, 255, 0, 255];
    let scale_offset: f32 = 0.8;
    let pos_offset = 0.8;
    let mut ornament_x = 0.6;
    let mut ornament_x_bottom = 0.2;
    let mut ornament_y = 0.0;
    let ornament_colors = [
        [255, 0, 0, 255],
        //[0, 150, 0, 255],
        [0, 0, 150, 255],
    ];

    let spawn_ornaments = false;

    for segment in 0..17 {
        for i in 0..6 {
            /*
            let starting_vel = [1.0, 1.6];
            let box_size = 10.0;

            let vel = [starting_vel[0] + random::<f32>() * 0.5, 0.0, starting_vel[1] + random::<f32>() * 0.5];
            let animation: Vec<Box<dyn ElementAnimation<Line>>> = vec![Box::new(SnowAnimation::new(
                2.0,
                Vector3::from(Position::new(vel)),
                0.0,
                [Position::new([-box_size, -box_size, -box_size]), Position::new([box_size, box_size, box_size])],
            ))];
            */
            let animation: Vec<Box<dyn ElementAnimation<Line>>> = vec![Box::new(RotateYAnimation::new(30.0))];
            renderer.objects.get_mut("tree").unwrap().elements.push(Box::new(Line::new(
                Vector3 { pos: verts[i]},
                Vector3 { pos:  verts[(i + 1) % 6]},
                line_color,
                animation.clone(),
            )));

            renderer.objects.get_mut("tree").unwrap().elements.push(Box::new(Line::new(
                Vector3 { pos: verts[i]},
                Vector3 { pos:  verts[i + 6]},
                line_color,
                animation.clone(),
            )));

            renderer.objects.get_mut("tree").unwrap().elements.push(Box::new(Line::new(
                Vector3 { pos: verts[i + 6]},
                Vector3 { pos:  verts[((i + 1) % 6) + 6]},
                line_color,
                animation.clone(),
            )));

            if spawn_ornaments {
                let ornament_fac = scale_offset.powi(segment);
                for _i in 0..(random::<f32>() * 4.0) as i32 {
                    let h = random::<f32>();
                    let rot = random::<f32>() * std::f32::consts::PI * 2.0;
                    let distance = ornament_x + (1.0 - h) * ornament_x_bottom;
                    let pos = [
                        f32::sin(rot) * distance,
                        h * pos_offset * ornament_fac + ornament_y,
                        f32::cos(rot) * distance,
                    ];
                    let ornament_anim: Vec<Box<dyn ElementAnimation<ParticleElement>>> = vec![Box::new(RotateYAnimation::new((random::<f32>() - 0.5) * 120.0))];

                    renderer.objects.get_mut("tree").unwrap().elements.push(Box::new(ParticleElement::new(
                        Position::new(pos),
                        10.0 * ornament_fac,
                        *ornament_colors.choose(&mut rand::thread_rng()).unwrap(),
                        ornament_anim,
                    )));
                }
            }
        }
        for i in 0..verts.len() {
            verts[i] = [verts[i][0] * scale_offset, verts[i][1] * scale_offset + pos_offset, verts[i][2] * scale_offset];
        }
        ornament_x *= scale_offset;
        ornament_x_bottom *= scale_offset;
        ornament_y = ornament_y * scale_offset + pos_offset;
        //println!("x: {}, bx: {}, y: {} [done]", ornament_x, ornament_x_bottom, ornament_y);
    }

    //star
    renderer.objects.get_mut("tree").unwrap().elements.push(Box::new(ParticleElement::new(
        Position::new([0.0, 4.0, 0.0]),
        10.0,
        [255, 255, 0, 255],
        vec![],
    )));

    //floor
    /*
    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line { start: RelativePosition { pos: [-2.0, 0.0, -2.0]}, end: RelativePosition { pos:  [-2.0, 0.0,  2.0]}, color: line_color}));
    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line { start: RelativePosition { pos: [ 2.0, 0.0, -2.0]}, end: RelativePosition { pos:  [ 2.0, 0.0,  2.0]}, color: line_color}));
    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line { start: RelativePosition { pos: [ 2.0, 0.0,  2.0]}, end: RelativePosition { pos:  [-2.0, 0.0,  2.0]}, color: line_color}));
    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line { start: RelativePosition { pos: [ 2.0, 0.0, -2.0]}, end: RelativePosition { pos:  [-2.0, 0.0, -2.0]}, color: line_color}));
    */

    //TEXT
    let text = Box::new(TextElement {
        position: ScreenPosition {
            pos: [0.0, 20.0],
            z: 1000.0,// render behind stuff
        },
        text: "Merry Crimas".to_string(),
        scale: 2.0,
        color: [255, 0, 0, 255],
        font: Font {
            font_image: font,
            width: 32,
            has_uppercase: false,
        }
    });
    renderer.objects.get_mut("text").unwrap().elements.push(text);

    //SNOW
    let starting_vel = [1.0, 1.6];
    let box_size = 12.0;
    for _i in 0..(if cfg!(target_arch = "wasm32") {500} else { 1500 }){
        let vel = [starting_vel[0] + random::<f32>() * 0.5, 0.0, starting_vel[1] + random::<f32>() * 0.5];
        let animation = Box::new(SnowAnimation::new(
            2.0,
            Position::new(vel),
            0.0,
            [Position::new([-box_size, -box_size, -box_size]), Position::new([box_size, box_size, box_size])],
        ));
        let pos = [
            (random::<f32>() - 0.5) * box_size * 2.0,
            (random::<f32>() - 0.5) * box_size * 2.0,
            (random::<f32>() - 0.5) * box_size * 2.0,
        ];
        renderer.objects.get_mut("tree").unwrap().elements.push(Box::new(ParticleElement::new(
            Vector3 { pos },
            4.0,
            [255, 255, 255, 255],
            //vec![animation, Box::new(RotateYAnimation::new(360.0))],
            vec![animation],
        )));
    }

    renderer.transforms.insert(0, Box::new(ScaleTransform::from([1.0, -1.0, 1.0])));
    renderer.transforms.insert(10, Box::new(XAxisRotation::from(15.0)));
    //renderer.transforms.insert("view_y_rot".to_string(), Box::new(XAxisRotation::from(15.0)));
    //renderer.transforms.insert("view_x_rot".to_string(), Box::new(XAxisRotation::from(15.0)));


    /*
    //cube
    renderer.lines.lines.push(Line::new([ 5.0, -5.0,  5.0], [ 5.0, -5.0, -5.0], [0, 255, 0, 255]));
    renderer.lines.lines.push(Line::new([ 5.0, -5.0,  5.0], [-5.0, -5.0,  5.0], [0, 255, 0, 255]));
    renderer.lines.lines.push(Line::new([-5.0, -5.0, -5.0], [ 5.0, -5.0, -5.0], [0, 255, 0, 255]));
    renderer.lines.lines.push(Line::new([-5.0, -5.0, -5.0], [-5.0, -5.0,  5.0], [0, 255, 0, 255]));

    renderer.lines.lines.push(Line::new([ 5.0,  5.0,  5.0], [ 5.0,  5.0, -5.0], [0, 255, 0, 255]));
    renderer.lines.lines.push(Line::new([ 5.0,  5.0,  5.0], [-5.0,  5.0,  5.0], [0, 255, 0, 255]));
    renderer.lines.lines.push(Line::new([-5.0,  5.0, -5.0], [ 5.0,  5.0, -5.0], [0, 255, 0, 255]));
    renderer.lines.lines.push(Line::new([-5.0,  5.0, -5.0], [-5.0,  5.0,  5.0], [0, 255, 0, 255]));

    renderer.lines.lines.push(Line::new([-5.0, -5.0, -5.0], [-5.0,  5.0, -5.0], [0, 255, 0, 255]));
    renderer.lines.lines.push(Line::new([-5.0, -5.0,  5.0], [-5.0,  5.0,  5.0], [0, 255, 0, 255]));
    renderer.lines.lines.push(Line::new([ 5.0, -5.0, -5.0], [ 5.0,  5.0, -5.0], [0, 255, 0, 255]));
    renderer.lines.lines.push(Line::new([ 5.0, -5.0,  5.0], [ 5.0,  5.0,  5.0], [0, 255, 0, 255]));
     */

    event_loop.set_control_flow(ControlFlow::Poll);

    let _ = event_loop.run(move |event, elwt| {
        //renderer.lines.lines.push(Line::new([-5.0, -5.0, -5.0], [5.0, 5.0, -5.0]));
        match event {
            Event::WindowEvent {
                window_id,
                event
            } => {
                if window_id == window.id() {
                    match event {
                        WindowEvent::MouseWheel {
                            delta, ..
                        } => {
                            //println!("mouse wheel delta: {:?}", delta);
                            match delta {
                                MouseScrollDelta::LineDelta(.., dy) => {
                                    renderer.scale += dy;
                                }
                                MouseScrollDelta::PixelDelta(delta) => {
                                    //println!("delta: {}, {}", delta.x, delta.y);

                                   if cfg!(target_arch = "wasm32") {
                                       renderer.scale += f32::clamp(delta.y as f32 / 15.0, -1.0, 1.0); //clamps because if scrolling with a mouse the value would be wayyy to high
                                   } else {
                                       renderer.scale += delta.y as f32 / 5.0
                                   };

                                }
                            }
                            renderer.scale = renderer.scale.clamp(8.0, 25.0);
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            let xrot = (position.y as f32 - window.inner_size().height as f32 / 2.0) / window.inner_size().height as f32 * 100.0;
                            let yrot = (position.x as f32 - window.inner_size().width as f32 / 2.0) / window.inner_size().width as f32 * 100.0;

                            renderer.transforms.insert(20, Box::new(YAxisRotation::from(-yrot)));
                            renderer.transforms.insert(21, Box::new(XAxisRotation::from(xrot)));
                        }
                        WindowEvent::KeyboardInput {  event, ..} => {
                            if event.state == ElementState::Pressed {
                                match event.logical_key {
                                    Key::Named(NamedKey::Escape) => {
                                        window.set_fullscreen(None);
                                        window.set_cursor_visible(true);
                                    }
                                    Key::Named(NamedKey::F11) => {
                                        window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                                        window.set_cursor_visible(false);
                                    }
                                    Key::Character(char) => {
                                        if char == "f" {
                                            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                                            window.set_cursor_visible(false);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let (width, height) = {
                                let size = window.inner_size();
                                (size.width, size.height)
                            };
                            surface
                                .resize(
                                    NonZeroU32::new(width).unwrap(),
                                    NonZeroU32::new(height).unwrap(),
                                )
                                .unwrap();

                            let mut buffer = surface.buffer_mut().unwrap();
                            //depth_buffer.fill(f32::INFINITY);
                            let mut depth_buffer = vec![f32::INFINITY; buffer.len()];
                            //depth_buffer.resize(buffer.len(), f32::INFINITY);
                            #[cfg(target_arch = "wasm32")]
                            if cfg!(target_arch = "wasm32"){
                                let web_window = web_sys::window().expect("no global `window` exists");
                                let (w, h) = (web_window.inner_width().unwrap().as_f64().unwrap() * web_window.device_pixel_ratio(), web_window.inner_height().unwrap().as_f64().unwrap() * web_window.device_pixel_ratio());
                                let _ = window.request_inner_size(PhysicalSize::new(w, h));
                            }

                            renderer.width = width as usize;
                            renderer.height = height as usize;

                            for element in renderer.objects.get_mut("text").unwrap().elements.iter_mut() {
                                element.set_size(renderer.width as f32 / 500.0);
                            }

                            renderer.update(Instant::now().duration_since(last_render));
                            last_render = Instant::now();

                            renderer.render(&mut buffer, &mut depth_buffer);
                            //renderer.draw_circle(&mut buffer, (50, 50), 10.0, [255, 255, 255, 255]);

                            //println!("{}", lines.lines.len());

                            buffer.present().unwrap();
                        }
                        WindowEvent::CloseRequested => {
                            elwt.exit();
                        }
                        _ => {}
                    }
                }
            }
            Event::NewEvents(..) => {
                if Instant::now().duration_since(last_frame) > Duration::from_secs_f64(1.0 / 60.0) {
                    renderer.rotation += 1.0;

                    last_frame = Instant::now();
                    window.request_redraw();
                } else {
                    #[cfg(not(target_arch = "wasm32"))] //this is broken on web, not really needed because of the poll strategy (see the top of main)
                    elwt.set_control_flow(ControlFlow::WaitUntil(last_frame.checked_add(Duration::from_secs_f64(1.0/60.0)).expect("wait time is wrong")));
                }
            }
            _ => {}
        }
    });

}
