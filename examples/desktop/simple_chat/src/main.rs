use anyhow::Result;
use chrono::{DateTime, Local};
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use woolly_core::{Engine, EngineConfig, GenerationConfig};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
    timestamp: DateTime<Local>,
}

struct ChatApp {
    messages: Vec<ChatMessage>,
    input: String,
    engine: Arc<Mutex<Option<Engine>>>,
    model_path: String,
    is_generating: bool,
    generation_config: GenerationConfig,
    runtime: tokio::runtime::Runtime,
}

impl Default for ChatApp {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            engine: Arc::new(Mutex::new(None)),
            model_path: String::new(),
            is_generating: false,
            generation_config: GenerationConfig::default(),
            runtime: tokio::runtime::Runtime::new().unwrap(),
        }
    }
}

impl ChatApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Load previous state if available
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }
        Self::default()
    }

    fn load_model(&mut self) {
        if self.model_path.is_empty() {
            return;
        }

        let engine = self.engine.clone();
        let model_path = self.model_path.clone();

        self.runtime.spawn(async move {
            let config = EngineConfig::default();
            match Engine::new(config) {
                Ok(mut eng) => {
                    if let Err(e) = eng.load_model(&model_path) {
                        eprintln!("Failed to load model: {}", e);
                    } else {
                        let mut engine_guard = engine.lock().await;
                        *engine_guard = Some(eng);
                        println!("Model loaded successfully");
                    }
                }
                Err(e) => eprintln!("Failed to create engine: {}", e),
            }
        });
    }

    fn send_message(&mut self, ctx: &egui::Context) {
        if self.input.trim().is_empty() || self.is_generating {
            return;
        }

        // Add user message
        let user_message = ChatMessage {
            role: "user".to_string(),
            content: self.input.clone(),
            timestamp: Local::now(),
        };
        self.messages.push(user_message);
        
        let prompt = self.build_prompt();
        self.input.clear();
        self.is_generating = true;

        // Generate response
        let engine = self.engine.clone();
        let config = self.generation_config.clone();
        let ctx = ctx.clone();

        self.runtime.spawn(async move {
            let mut engine_guard = engine.lock().await;
            if let Some(engine) = engine_guard.as_mut() {
                match engine.generate(&prompt, &config).await {
                    Ok(response) => {
                        ctx.request_repaint();
                        drop(engine_guard);
                        
                        // Will be handled in the next frame
                        println!("Generated response: {}", response);
                    }
                    Err(e) => {
                        eprintln!("Generation error: {}", e);
                    }
                }
            }
        });
    }

    fn build_prompt(&self) -> String {
        let mut prompt = String::new();
        
        for message in &self.messages {
            match message.role.as_str() {
                "user" => prompt.push_str(&format!("User: {}\n", message.content)),
                "assistant" => prompt.push_str(&format!("Assistant: {}\n", message.content)),
                _ => {}
            }
        }
        
        prompt.push_str("Assistant: ");
        prompt
    }
}

impl eframe::App for ChatApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top panel for settings
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Woolly Chat");
                
                ui.separator();
                
                ui.label("Model:");
                if ui.text_edit_singleline(&mut self.model_path).changed() {
                    // Model path changed
                }
                
                if ui.button("Load Model").clicked() {
                    self.load_model();
                }
                
                ui.separator();
                
                // Model status
                let status = self.runtime.block_on(async {
                    self.engine.lock().await.is_some()
                });
                
                if status {
                    ui.colored_label(egui::Color32::GREEN, "Model Loaded");
                } else {
                    ui.colored_label(egui::Color32::RED, "No Model");
                }
            });
        });

        // Bottom panel for input
        egui::TopBottomPanel::bottom("bottom_panel")
            .min_height(60.0)
            .show(ctx, |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        let response = ui.add(
                            egui::TextEdit::multiline(&mut self.input)
                                .desired_width(ui.available_width() - 100.0)
                                .desired_rows(2)
                                .hint_text("Type your message...")
                        );

                        if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            self.send_message(ctx);
                        }

                        if ui.button("Send").clicked() {
                            self.send_message(ctx);
                        }
                    });

                    // Generation settings
                    ui.collapsing("Settings", |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Max Tokens:");
                            ui.add(egui::DragValue::new(&mut self.generation_config.max_tokens)
                                .speed(10)
                                .clamp_range(1..=2048));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Temperature:");
                            ui.add(egui::Slider::new(&mut self.generation_config.temperature, 0.0..=2.0));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Top P:");
                            ui.add(egui::Slider::new(&mut self.generation_config.top_p, 0.0..=1.0));
                        });
                    });
                });
            });

        // Central panel for chat
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    for message in &self.messages {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                match message.role.as_str() {
                                    "user" => {
                                        ui.colored_label(egui::Color32::from_rgb(100, 149, 237), "You");
                                    }
                                    "assistant" => {
                                        ui.colored_label(egui::Color32::from_rgb(50, 205, 50), "AI");
                                    }
                                    _ => {}
                                }
                                
                                ui.label(format!("[{}]", message.timestamp.format("%H:%M:%S")));
                            });
                            
                            ui.label(&message.content);
                        });
                        
                        ui.add_space(5.0);
                    }

                    if self.is_generating {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Generating response...");
                        });
                    }
                });
        });
    }
}

fn main() -> Result<()> {
    // Setup logging
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Woolly Chat",
        options,
        Box::new(|cc| Box::new(ChatApp::new(cc))),
    ).map_err(|e| anyhow::anyhow!("Failed to run app: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_building() {
        let app = ChatApp::default();
        let prompt = app.build_prompt();
        assert_eq!(prompt, "Assistant: ");
    }
}