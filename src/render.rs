use std::sync::Arc;

use glam::*;


pub struct Resources {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    window: Arc<winit::window::Window>,
    depth_texture: Option<wgpu::Texture>,
}

impl Resources {
    fn init(window: Arc<winit::window::Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());

        let surface = instance.create_surface(window.clone())?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptionsBase {
            power_preference: wgpu::PowerPreference::from_env().unwrap_or_default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))?;

        let experimental_flags = wgpu::Features {
            features_wgpu: (
                wgpu::FeaturesWGPU::EXPERIMENTAL_MESH_SHADER |
                wgpu::FeaturesWGPU::EXPERIMENTAL_MESH_SHADER_MULTIVIEW |
                wgpu::FeaturesWGPU::EXPERIMENTAL_PASSTHROUGH_SHADERS |
                wgpu::FeaturesWGPU::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN |
                wgpu::FeaturesWGPU::EXPERIMENTAL_RAY_QUERY 
            ),
            features_webgpu: wgpu::FeaturesWebGPU::empty(),
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { 
            label: None, 
            required_features: adapter.features().difference(experimental_flags), 
            required_limits: adapter.limits(), 
            ..Default::default()
        }))?;

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(surface_caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let mut this = Self { 
            device: Arc::new(device), 
            queue: Arc::new(queue), 
            surface, 
            surface_config, 
            window,
            depth_texture: None
        };

        this.resize(size);

        Ok(this)
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.surface_config.width = size.width;
        self.surface_config.height = size.height;
        self.surface.configure(&self.device, &self.surface_config);
        self.depth_texture.replace(self.device.create_texture(&wgpu::TextureDescriptor { 
            label: None, 
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            }, 
            mip_level_count: 1, 
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::Depth32Float, 
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, 
            view_formats: &[]
        }));
    }

    pub fn surface_texture_size(&self) -> UVec2 {
        uvec2(self.surface_config.width, self.surface_config.height)
    }

    pub fn surface_texture_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    pub fn depth_texture_format(&self) -> wgpu::TextureFormat {
        self.depth_texture.as_ref().unwrap().format()
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
}

#[derive(Default)]
pub struct Renderer {
    resources: Option<Resources>,
    init_func: Option<Box<dyn FnOnce(&mut Self) -> anyhow::Result<()>>>,
    update_func: Option<Box<dyn FnMut(&mut Self) -> anyhow::Result<()>>>,
    render_func: Option<Box<dyn FnMut(&mut Self, RenderCtx) -> anyhow::Result<()>>>,


    scale_factor: f64,
    old_cursor_pos: winit::dpi::LogicalPosition<f64>,
    cursor_pos: winit::dpi::LogicalPosition<f64>,
    scroll_delta: winit::dpi::LogicalPosition<f64>,
    old_mouse_buttons: std::collections::HashMap<MouseButton, bool>,
    mouse_buttons: std::collections::HashMap<MouseButton, bool>,
}

pub struct RenderCtx<'a> {
    pub command_encoder: &'a mut wgpu::CommandEncoder,
    pub surface_texture_view: &'a wgpu::TextureView,
    pub depth_texture_view: &'a wgpu::TextureView,
}

pub use winit::event::MouseButton;

impl Renderer {
    pub fn resources(&self) -> &Resources {
        self.resources.as_ref().unwrap()
    }

    pub fn resources_mut(&mut self) -> &mut Resources {
        self.resources.as_mut().unwrap()
    }

    pub fn window_size(&self) -> Vec2 {
        let size = self.resources().window.inner_size().to_logical(self.scale_factor);
        vec2(size.width, size.height)
    }

    pub fn cursor_pos(&self) -> Vec2 {
        let pos = self.cursor_pos.cast();
        vec2(pos.x, pos.y)
    }

    fn old_cursor_pos(&self) -> Vec2 {
        let pos = self.old_cursor_pos.cast();
        vec2(pos.x, pos.y)
    }

    pub fn cursor_delta(&self) -> Vec2 {
        self.cursor_pos() - self.old_cursor_pos()
    }

    pub fn scroll_delta(&self) -> Vec2 {
        let delta = self.scroll_delta.cast();
        vec2(delta.x, delta.y)
    }

    pub fn mouse_button_state(&self, button: MouseButton) -> bool {
        self.mouse_buttons.get(&button).copied().unwrap_or(false)
    }

    pub fn set_update_callback<F: FnMut(&mut Self) -> anyhow::Result<()> + 'static>(&mut self, f: F) {
        self.update_func.replace(Box::new(f));
    }

    pub fn set_render_callback<F: FnMut(&mut Self, RenderCtx) -> anyhow::Result<()> + 'static>(&mut self, f: F) {
        self.render_func.replace(Box::new(f));
    }

    fn on_resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) -> anyhow::Result<()> {
        if self.resources.is_none() {
            let window = Arc::new(event_loop.create_window(Default::default())?);
            self.resources.replace(Resources::init(window)?);
            self.scale_factor = 1.0;
        }

        if let Some(init_fn) = self.init_func.take() {
            init_fn(self)?;
        }

        Ok(())
    }

    fn on_redraw(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) -> anyhow::Result<()> {
        self.resources().window.request_redraw();
        if let Some(mut update_func) = self.update_func.take() {
            update_func(self)?;
            if self.update_func.is_none() {
                self.update_func.replace(update_func);
            }
        }
        if let Some(mut render_func) = self.render_func.take() {
            let surface_texture = self.resources().surface.get_current_texture()?;
            let surface_texture_view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_texture_view = self.resources().depth_texture.as_ref().unwrap().create_view(&wgpu::wgt::TextureViewDescriptor::default());

            let mut command_encoder = self.resources().device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());


            render_func(self, RenderCtx { 
                command_encoder: &mut command_encoder, 
                surface_texture_view: &surface_texture_view, 
                depth_texture_view: &depth_texture_view 
            })?;

            self.resources().queue().submit(Some(command_encoder.finish()));
            surface_texture.present();

            if self.render_func.is_none() {
                self.render_func.replace(render_func);
            }
        }
        self.scroll_delta = winit::dpi::LogicalPosition::new(0.0, 0.0);
        self.old_cursor_pos = self.cursor_pos;
        self.old_mouse_buttons.extend(self.mouse_buttons.iter().map(|(mb, s)| (*mb, *s)));
        Ok(())
    }
}

impl winit::application::ApplicationHandler for Renderer {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.on_resumed(event_loop).unwrap();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if self.resources.is_none() { return; }

        match event {
            winit::event::WindowEvent::RedrawRequested => {
                self.on_redraw(event_loop).unwrap()
            },

            winit::event::WindowEvent::Moved(physical_position) => {},
            winit::event::WindowEvent::Resized(size) => {
                self.resources_mut().resize(size);
            },
            winit::event::WindowEvent::ScaleFactorChanged { scale_factor, inner_size_writer } => {
                self.scale_factor = scale_factor;
            },
            winit::event::WindowEvent::CloseRequested => {
                event_loop.exit();
            },
            winit::event::WindowEvent::Destroyed => {},
            winit::event::WindowEvent::Occluded(_) => {},
            
            winit::event::WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {},
            winit::event::WindowEvent::MouseInput { device_id, state, button } => {
                self.mouse_buttons.insert(button, state.is_pressed());
            },
            winit::event::WindowEvent::MouseWheel { device_id, delta, phase } => {
                let delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => winit::dpi::LogicalPosition::new(x as f64, y as f64),
                    winit::event::MouseScrollDelta::PixelDelta(delta) => delta.to_logical(self.scale_factor),
                };
                self.scroll_delta.x += delta.x;
                self.scroll_delta.y += delta.y;
            },
            winit::event::WindowEvent::CursorMoved { device_id, position } => {
                let position = position.to_logical(self.scale_factor);
                self.cursor_pos = position;
            },
            winit::event::WindowEvent::CursorEntered { device_id } => {},
            winit::event::WindowEvent::CursorLeft { device_id } => {},

            _ => {}
        }
    }
}

pub fn run<F: FnOnce(&mut Renderer) -> anyhow::Result<()> + 'static>(init_fn: F) -> ! {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut renderer = Renderer::default();
    renderer.init_func = Some(Box::new(init_fn));
    event_loop.run_app(&mut renderer).unwrap();
    std::process::exit(0)
}
