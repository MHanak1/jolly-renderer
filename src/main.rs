#![windows_subsystem = "windows"]

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::ops::{Deref, Mul};
use std::rc::Rc;
use std::time::{Duration, Instant};
use winit::event::{Event, MouseScrollDelta, StartCause, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use dyn_clone::DynClone;
use image::{GenericImage, Pixel, RgbaImage};
use rand::random;

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
        &mut self.pos[2]
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
        for i in 0..3 {
            new_pos.pos[i] = self.pos[i] + offset[i];
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
        let rad = self.angle / 180.0 * 3.14;
        Matrix3 {
            mat: [[1.0, 0.0, 0.0],
                [0.0, f32::cos(rad), -f32::sin(rad)],
                [0.0, f32::sin(rad), f32::cos(rad)]]
        }
    }
}

impl TransformMatrix for YAxisRotation {
    fn get_rotation_matrix(&self) -> Matrix3<f32> {
        let rad = self.angle / 180.0 * 3.14;
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
        let rad = self.angle / 180.0 * 3.14;
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
            char_index = char_index - 6;
        }
        if !self.has_uppercase {
            char_index = char_index % 26;
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

trait Element: DynClone {
    fn update(&mut self, dt: Duration);
    fn render(&self, renderer: &Renderer, buffer: &mut [u32]);
    fn transform_by(&mut self, matrix: Matrix3<f32>);
    fn scale_by(&mut self, scalar: f32);
    fn set_size(&mut self, size: f32);
    fn offset(&mut self, offset: [f32; 3]);
    fn get_pos(&self) -> Vector3;
    fn get_depth(&self) -> f32;
}
dyn_clone::clone_trait_object!(Element);

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

impl Element for Line {
    //TODO: unfuck this code (i mean it works but it really hurts to look at)
    fn update(&mut self, time_delta: Duration) {
        let mut new = self.clone();
        for animation in self.animations.iter_mut() {
            let new1 = animation.update(new.clone(), time_delta);
            new = new1;
        }
        *self = new;
    }

    fn render(&self, renderer: &Renderer, buffer: &mut [u32]){
        let line_start = renderer.get_pixel_pos([self.start.x(), self.start.y()]);
        let line_end = renderer.get_pixel_pos([self.end.x(), self.end.y()]);
        renderer.draw_line(buffer, (line_start[0], line_start[1]), (line_end[0], line_end[1]), self.color);
    }

    fn transform_by(&mut self, matrix: Matrix3<f32>) {
        self.start = self.start.mul(matrix.clone());
        self.end = self.end.mul(matrix.clone());
    }

    fn scale_by(&mut self, scalar: f32) {
        self.start = self.start.scale(scalar);
        self.end = self.end.scale(scalar);
    }

    fn set_size(&mut self, size: f32) {
        self.width = size;
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

impl Element for TextElement {
    fn update(&mut self, time_delta: Duration) {
        //do nothing
    }

    fn render(&self, renderer: &Renderer, buffer: &mut [u32]){
        let scale = i32::clamp(self.scale as i32, 1, i32::max_value());
        for character in 0..self.text.len() {
            let img = self.font.get_character(self.text.chars().nth(character).unwrap());
            if let Some(img) = img {
                for x in 0..img.width() {
                    for y in 0..img.height() {
                        let color = img.get_pixel(x, y).to_rgba();
                        for i in 0..scale{
                            for j in 0..scale {
                                renderer.set_color_with_alpha(
                                    buffer,
                                    (x as i32 * scale
                                         + character as i32 * img.height() as i32 * scale
                                         + self.position.get()[0] as i32
                                         - self.text.len() as i32 * self.font.width as i32 / 2 * scale
                                         + renderer.width as i32 / 2
                                         + i,
                                     y as i32 * scale
                                         + self.position.get()[1] as i32
                                         + j),
                                    color.0
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    fn transform_by(&mut self, matrix: Matrix3<f32>) {
        //self.position = self.position.mul(matrix);
    }

    fn scale_by(&mut self, scalar: f32) {
        self.scale = self.scale.mul(scalar);
    }

    fn set_size(&mut self, size: f32) {
        self.scale = size;
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

impl Element for ParticleElement {
    fn update(&mut self, time_delta: Duration) {
        let mut new = self.clone();
        for animation in self.animations.iter_mut() {
            let new1 = animation.update(new.clone(), time_delta);
            new = new1;
        }
        *self = new;
    }

    fn render(&self, renderer: &Renderer, buffer: &mut [u32]){
        let pos = renderer.get_pixel_pos([self.position.x(), self.position.y()]);
        renderer.draw_circle(buffer, (pos[0] as i32, pos[1] as i32), self.size * renderer.get_render_scale(), self.color)
    }

    fn transform_by(&mut self, matrix: Matrix3<f32>) {
        self.position = self.position.mul(matrix.clone());
    }

    fn scale_by(&mut self, scalar: f32) {
        self.position = self.position.scale(scalar);
    }

    fn set_size(&mut self, size: f32) {
        self.size = size;
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
trait ElementAnimation<T: Element>: DynClone{
    fn update(&mut self, element: T, dt: Duration) -> T;
}
dyn_clone::clone_trait_object!(<T> ElementAnimation<T> where T: Element);

#[derive(Clone)]
struct RotateYAnimation {
    rotation: f32,
    rate: f32,
}

impl RotateYAnimation {
    fn new(rate: f32) -> Self {
        Self { rotation: 0.0, rate }
    }
}

impl<T: Element + DynClone + Clone> ElementAnimation<T> for RotateYAnimation {
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
}

impl SnowAnimation {
    fn new(fall_speed: f32, velocity: Vector3, velocity_noise: f32, bounds: [Vector3; 2]) -> SnowAnimation {
        SnowAnimation {
            fall_speed,
            velocity,
            velocity_noise,
            bounds,
        }
    }
}

impl<T: Element + DynClone> ElementAnimation<T> for SnowAnimation {
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

        new.offset(self.velocity.scale(dt.as_secs_f32()).pos);
        for i in 0..3{
            let (lbound, hbound) =
                (f32::min(self.bounds[0].pos[i], self.bounds[1].pos[i]),
                 f32::max(self.bounds[0].pos[i], self.bounds[1].pos[i]));

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
        new
    }
}

struct ElementContainer {
    transforms: HashMap<String, Box<dyn TransformMatrix>>,
    position: Box<dyn Position>,
    elements: Vec<Box<dyn Element>>,
}

impl ElementContainer {
    fn empty() -> ElementContainer {
        ElementContainer {
            transforms: HashMap::new(),
            position: Box::<Vector3>::new(Vector3::default()),
            elements: Vec::new()
        }
    }
    fn get_transformed_elements(&self) -> Vec<Box<dyn Element>>{
        let mut new_elements : Vec<Box<dyn Element>> = Vec::new();
        for element in self.elements.iter() {
            let mut new_element = element.clone();
            new_element.offset(self.position.get());
            for matrix in self.transforms.values() {
                new_element.transform_by(matrix.get_rotation_matrix())
            }
            new_elements.push(new_element);
        }
        new_elements
    }
}



struct Renderer {
    width: usize,
    height: usize,
    scale: f32,
    containers: HashMap<String, ElementContainer>,
    rotation: f32,
    position_offset: [f32; 2],
    transforms: HashMap<String, Box<dyn TransformMatrix>>,
}

impl Renderer{
    fn new(width: usize, height: usize) -> Renderer {
        Renderer {
            width,
            height,
            scale: 11.0,
            containers: HashMap::new(),
            rotation: 0.0,
            position_offset: [0.0; 2],
            transforms: HashMap::new(),
        }
    }

    fn render (&self, buffer: &mut [u32]) {
        buffer.fill(0);
        for container in self.containers.values() {
            for element in container.get_transformed_elements().iter() {
                let mut element = element.clone();
                for matrix in self.transforms.values() {
                    element.transform_by(matrix.get_rotation_matrix())
                }
                element.render(self, buffer);
            }
        }
    }

    fn update (&mut self, time_delta: Duration) {
        for container in self.containers.values_mut() {
            for element in container.elements.iter_mut() {
                element.update(time_delta);
            }
        }
    }

    fn get_render_scale (&self) -> f32 {
        self.scale * self.scale / 100.0
    }

    fn draw_line (&self, buffer: &mut [u32], (x_start, y_start): (f32, f32), (x_end, y_end): (f32, f32), color: [u8; 4]) {
//stolen from wikipedia
        let (mut x, mut y) = (x_start, y_start);
        let dx = f32::abs(f32::clamp((x_end - x) / (y_end - y), -1.0, 1.0)) * if x_start > x_end {-1.0} else {1.0};
        let dy = f32::abs(f32::clamp((y_end - y) / (x_end - x), -1.0, 1.0)) * if y_start > y_end {-1.0} else {1.0};

        let mut iters = 0;

        while f32::max(f32::abs(x - x_start), f32::abs(y - y_start)) < f32::max(f32::abs(y_start - y_end), f32::abs(x_start - x_end)) {
            self.set_color_with_alpha(buffer, (x as i32, y as i32), color);
            iters += 1;

            if iters > 10000 {
                println!("WARN: Too many line draw iterations. Aborting. (start: ({}, {}), end: ({}, {}), delta x: {}, delta y: {})", x_start, y_start, x_end, y_end, dx, dy);
                return()
            }

            //x += if right {1.0} else {-1.0};
            x += dx;
            y += dy;
        }
    }

    fn draw_circle(&self, buffer: &mut [u32], (x, y): (i32, i32), radius: f32, color: [u8; 4]) {
        for i in (-radius.floor() as i32)..(radius.ceil() as i32) {
            for j in (-radius.floor() as i32)..(radius.ceil() as i32) {
                if ((i * i + j * j) as f32) < radius * radius {
                    self.set_color_with_alpha(buffer, (x + i, y  + j), color);
                }
            }
        }
    }

    fn get_pixel_pos(&self, pos: [f32; 2]) -> [f32; 2] {
        [(self.width / 2) as f32 + (pos[0] * self.get_render_scale() * 100.0) + self.position_offset[0], (self.height / 2) as f32 + (pos[1] * self.get_render_scale() * 100.0) + self.position_offset[1]]
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
            ((color[2] as u32) <<  0);
    }

    fn get_color (&self, buffer: &[u32], (x, y): (usize, usize)) -> [u8; 3]{
        let col: [u8; 4] = buffer[x + y * self.width].to_be_bytes();
        [col[1], col[2], col[3]]

    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let window = Rc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let context = softbuffer::Context::new(window.clone()).unwrap();
    let mut surface = softbuffer::Surface::new(&context, window.clone()).unwrap();

    let mut renderer = Renderer::new(window.inner_size().width as usize, window.inner_size().height as usize);
    let mut lastRender = Instant::now();

    let font_bytes = include_bytes!("resources/font.png");

    let font = image::load_from_memory(font_bytes).unwrap().to_rgba8();

    //vertices
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

    renderer.containers.insert("tree".to_string(), ElementContainer::empty());
    let line_color = [0, 255, 0, 255];
    let scale_offset = 0.8;
    let pos_offset = 0.8;

    for _ in 0..20 {
        for i in 0..6 {
            let animation: Vec<Box<dyn ElementAnimation<Line>>> = vec![Box::new(RotateYAnimation::new(30.0))];
            renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line::new(
                Vector3 { pos: verts[i]},
                Vector3 { pos:  verts[(i + 1) % 6]},
                line_color,
                animation.clone(),
            )));

            renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line::new(
                Vector3 { pos: verts[i]},
                Vector3 { pos:  verts[i + 6]},
                line_color,
                animation.clone(),
            )));

            renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line::new(
                Vector3 { pos: verts[i + 6]},
                Vector3 { pos:  verts[((i + 1) % 6) + 6]},
                line_color,
                animation.clone(),
            )));
        }
        for i in 0..verts.len() {
            verts[i] = [verts[i][0] * scale_offset, verts[i][1] * scale_offset + pos_offset, verts[i][2] * scale_offset]
        }
    }
    //floor
    /*
    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line { start: RelativePosition { pos: [-2.0, 0.0, -2.0]}, end: RelativePosition { pos:  [-2.0, 0.0,  2.0]}, color: line_color}));
    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line { start: RelativePosition { pos: [ 2.0, 0.0, -2.0]}, end: RelativePosition { pos:  [ 2.0, 0.0,  2.0]}, color: line_color}));
    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line { start: RelativePosition { pos: [ 2.0, 0.0,  2.0]}, end: RelativePosition { pos:  [-2.0, 0.0,  2.0]}, color: line_color}));
    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(Line { start: RelativePosition { pos: [ 2.0, 0.0, -2.0]}, end: RelativePosition { pos:  [-2.0, 0.0, -2.0]}, color: line_color}));
    */
    renderer.containers.get_mut("tree").unwrap().position = Box::<Vector3>::new(Vector3::new([0.0, -2.0, 0.0]));
    //renderer.containers.get_mut("tree").unwrap().position = Box::<Vector3>::new(Vector3::new([0.0, -1.2, 0.0]));

    renderer.containers.insert("text".to_string(), ElementContainer::empty());
    let text = Box::new(TextElement {
        position: ScreenPosition {
            pos: [0.0, 20.0],
            z: 0.0,
        },
        text: "Merry Christmas".to_string(),
        scale: 2.0,
        color: [255, 255, 255, 255],
        font: Font {
            font_image: font,
            width: 32,
            has_uppercase: false,
        }
    });
    renderer.containers.get_mut("text").unwrap().elements.push(text);

    renderer.containers.insert("snow".to_string(), ElementContainer::empty());

    let starting_vel = [1.0, 1.6];
    let box_size = 10.0;

    for i in 0..500 {
        let vel = [starting_vel[0] + random::<f32>() * 0.5, 0.0, starting_vel[1] + random::<f32>() * 0.5];
        let animation = Box::new(SnowAnimation::new(
            2.0,
            Vector3::from(Position::new(vel)),
            0.0,
            [Position::new([-box_size, -box_size, -box_size]), Position::new([box_size, box_size, box_size])],
        ));
        let pos = [
            (random::<f32>() - 0.5) * box_size * 2.0,
            (random::<f32>() - 0.5) * box_size * 2.0,
            (random::<f32>() - 0.5) * box_size * 2.0,
        ];
        renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(ParticleElement::new(
            Vector3 { pos },
            4.0,
            [255, 255, 255, 255],
            vec![animation],
        )));
    }

    /*
    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(ParticleElement::new(
        Vector3 { pos: [1.0, 1.0, 0.0] },
        10.0,
        [255, 0, 0, 255],
        vec![animation.clone()],
    )));

    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(ParticleElement::new(
        Vector3 { pos: [-1.0, 1.0, 0.0] },
        10.0,
        [255, 0, 0, 255],
        vec![animation.clone()],
    )));

    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(ParticleElement::new(
        Vector3 { pos: [0.0, 1.0, 1.0] },
        10.0,
        [0, 0, 255, 255],
        vec![animation.clone()],
    )));

    renderer.containers.get_mut("tree").unwrap().elements.push(Box::new(ParticleElement::new(
        Vector3 { pos: [0.0, 1.0, -1.0] },
        10.0,
        [0, 0, 255, 255],
        vec![animation.clone()]
    )));
     */

    renderer.transforms.insert("y_flip".to_string(), Box::new(ScaleTransform::from([1.0, -1.0, 1.0])));
    renderer.transforms.insert("x_rot_bias".to_string(), Box::new(XAxisRotation::from(15.0)));
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

    event_loop.set_control_flow(ControlFlow::WaitUntil(
        Instant::now().checked_add(Duration::from_millis(1000/60)).unwrap(),
    ));

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
                                    renderer.scale += delta.y as f32 / 5.0;

                                }
                            }
                            renderer.scale = renderer.scale.clamp(8.0, 25.0);
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            let xrot = (position.y as f32 - renderer.height as f32 / 2.0) / 10.0;
                            let yrot = (position.x as f32 - renderer.width as f32 / 2.0) / 10.0;

                            //println!("x: {}, t: {}", xrot, yrot);
                            renderer.transforms.insert("view_x_rot".parse().unwrap(), Box::new(XAxisRotation::from(xrot)));
                            renderer.transforms.insert("view_y_rot".parse().unwrap(), Box::new(YAxisRotation::from (yrot)));
                            //println!("x: {}, t: {}", xrot, yrot);
                            //renderer.transforms.insert("view_x_rot".parse().unwrap(), Box::new(YAxisRotation {angle: 0.0}));
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
                            renderer.width = width as usize;
                            renderer.height = height as usize;

                            for element in renderer.containers.get_mut("text").unwrap().elements.iter_mut() {
                                element.set_size(renderer.width as f32 / 500.0);
                            }

                            renderer.update(Instant::now().duration_since(lastRender));
                            lastRender = Instant::now();

                            renderer.render(&mut buffer);
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
            Event::NewEvents(cause) => {
                match cause {
                    StartCause::ResumeTimeReached { .. } => {
                        elwt.set_control_flow(ControlFlow::WaitUntil(
                            Instant::now().checked_add(Duration::from_millis(1000/60)).unwrap(),
                        ));
                        renderer.rotation += 1.0;
                        //renderer.containers.get_mut("tree").unwrap().y_rot_matrix = get_y_rotation_matrix(renderer.rotation);

                        window.request_redraw();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    });

}
