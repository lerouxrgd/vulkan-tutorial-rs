const WIDTH: u32 = 1080;
const HEIGHT: u32 = 1080;

struct HelloTriangleApp {
    sdl_context: sdl3::Sdl,
    window: sdl3::video::Window,
}

impl HelloTriangleApp {
    pub fn new() -> anyhow::Result<Self> {
        let sdl_context = sdl3::init()?;
        let video = sdl_context.video()?;
        let window: sdl3::video::Window = video
            .window("Vulkan tutorial", WIDTH, HEIGHT)
            .vulkan()
            .resizable()
            .build()?;
        Ok(Self {
            sdl_context,
            window,
        })
    }

    fn run(self) -> anyhow::Result<()> {
        let mut event_pump = self.sdl_context.event_pump()?;
        let mut quit = false;
        while !quit {
            for event in event_pump.poll_iter() {
                match event {
                    sdl3::event::Event::Quit { .. } => {
                        quit = true;
                        break;
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let app = HelloTriangleApp::new()?;
    app.run()?;
    Ok(())
}
