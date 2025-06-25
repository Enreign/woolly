use anyhow::Result;
use arboard::Clipboard;
use futures::StreamExt;
use iced::widget::{button, column, container, row, scrollable, text, text_editor, text_input};
use iced::{executor, Application, Command, Element, Length, Settings, Theme};
use std::path::PathBuf;
use std::sync::Arc;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style, ThemeSet};
use syntect::parsing::SyntaxSet;
use syntect::util::LinesWithEndings;
use tokio::sync::Mutex;
use woolly_core::{Engine, EngineConfig, GenerationConfig};

pub fn main() -> iced::Result {
    CodeAssistant::run(Settings::default())
}

#[derive(Debug, Clone)]
enum Message {
    CodeChanged(text_editor::Action),
    LanguageChanged(String),
    ModelPathChanged(String),
    LoadModel,
    ModelLoaded(Result<String, String>),
    
    // Actions
    ExplainCode,
    ImproveCode,
    FindBugs,
    GenerateTests,
    AddComments,
    
    // Results
    ResponseReceived(String),
    CopyToClipboard(String),
    
    // UI
    TabSelected(Tab),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tab {
    Editor,
    Response,
    Settings,
}

struct CodeAssistant {
    // Editor state
    code_content: text_editor::Content,
    language: String,
    syntax_set: Arc<SyntaxSet>,
    theme_set: Arc<ThemeSet>,
    
    // Model state
    engine: Arc<Mutex<Option<Engine>>>,
    model_path: String,
    model_loaded: bool,
    is_generating: bool,
    
    // Response
    response: String,
    
    // UI state
    active_tab: Tab,
    
    // Clipboard
    clipboard: Option<Clipboard>,
}

impl Application for CodeAssistant {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let syntax_set = Arc::new(SyntaxSet::load_defaults_newlines());
        let theme_set = Arc::new(ThemeSet::load_defaults());
        
        (
            Self {
                code_content: text_editor::Content::new(),
                language: "rust".to_string(),
                syntax_set,
                theme_set,
                engine: Arc::new(Mutex::new(None)),
                model_path: String::new(),
                model_loaded: false,
                is_generating: false,
                response: String::new(),
                active_tab: Tab::Editor,
                clipboard: Clipboard::new().ok(),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("Woolly Code Assistant")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::CodeChanged(action) => {
                self.code_content.perform(action);
                Command::none()
            }
            
            Message::LanguageChanged(lang) => {
                self.language = lang;
                Command::none()
            }
            
            Message::ModelPathChanged(path) => {
                self.model_path = path;
                Command::none()
            }
            
            Message::LoadModel => {
                self.model_loaded = false;
                let model_path = self.model_path.clone();
                let engine = self.engine.clone();
                
                Command::perform(
                    async move {
                        let config = EngineConfig::default();
                        match Engine::new(config) {
                            Ok(mut eng) => {
                                if let Err(e) = eng.load_model(&model_path) {
                                    Err(format!("Failed to load model: {}", e))
                                } else {
                                    let mut engine_guard = engine.lock().await;
                                    *engine_guard = Some(eng);
                                    Ok(model_path)
                                }
                            }
                            Err(e) => Err(format!("Failed to create engine: {}", e)),
                        }
                    },
                    Message::ModelLoaded,
                )
            }
            
            Message::ModelLoaded(result) => {
                match result {
                    Ok(_) => {
                        self.model_loaded = true;
                        self.response = "Model loaded successfully!".to_string();
                    }
                    Err(e) => {
                        self.response = format!("Error: {}", e);
                    }
                }
                self.active_tab = Tab::Response;
                Command::none()
            }
            
            Message::ExplainCode => {
                self.generate_with_prompt("Explain the following code:\n\n")
            }
            
            Message::ImproveCode => {
                self.generate_with_prompt("Improve the following code:\n\n")
            }
            
            Message::FindBugs => {
                self.generate_with_prompt("Find potential bugs in the following code:\n\n")
            }
            
            Message::GenerateTests => {
                self.generate_with_prompt("Generate unit tests for the following code:\n\n")
            }
            
            Message::AddComments => {
                self.generate_with_prompt("Add helpful comments to the following code:\n\n")
            }
            
            Message::ResponseReceived(response) => {
                self.response = response;
                self.is_generating = false;
                self.active_tab = Tab::Response;
                Command::none()
            }
            
            Message::CopyToClipboard(text) => {
                if let Some(clipboard) = &mut self.clipboard {
                    let _ = clipboard.set_text(text);
                }
                Command::none()
            }
            
            Message::TabSelected(tab) => {
                self.active_tab = tab;
                Command::none()
            }
        }
    }

    fn view(&self) -> Element<Message> {
        let tabs = row![
            button("Editor")
                .on_press(Message::TabSelected(Tab::Editor))
                .style(if self.active_tab == Tab::Editor {
                    button::primary
                } else {
                    button::secondary
                }),
            button("Response")
                .on_press(Message::TabSelected(Tab::Response))
                .style(if self.active_tab == Tab::Response {
                    button::primary
                } else {
                    button::secondary
                }),
            button("Settings")
                .on_press(Message::TabSelected(Tab::Settings))
                .style(if self.active_tab == Tab::Settings {
                    button::primary
                } else {
                    button::secondary
                }),
        ]
        .spacing(10);

        let content = match self.active_tab {
            Tab::Editor => self.view_editor(),
            Tab::Response => self.view_response(),
            Tab::Settings => self.view_settings(),
        };

        let actions = row![
            button("Explain").on_press(Message::ExplainCode),
            button("Improve").on_press(Message::ImproveCode),
            button("Find Bugs").on_press(Message::FindBugs),
            button("Generate Tests").on_press(Message::GenerateTests),
            button("Add Comments").on_press(Message::AddComments),
        ]
        .spacing(10);

        column![tabs, actions, content]
            .spacing(20)
            .padding(20)
            .into()
    }
}

impl CodeAssistant {
    fn view_editor(&self) -> Element<Message> {
        let editor = text_editor(&self.code_content)
            .on_action(Message::CodeChanged)
            .height(Length::Fill)
            .font(iced::Font::MONOSPACE);

        let language_selector = row![
            text("Language:"),
            text_input("rust", &self.language)
                .on_input(Message::LanguageChanged)
                .width(100),
        ]
        .spacing(10);

        column![language_selector, editor]
            .spacing(10)
            .into()
    }

    fn view_response(&self) -> Element<Message> {
        let response_view = scrollable(
            column![
                text(&self.response).font(iced::Font::MONOSPACE),
                button("Copy to Clipboard")
                    .on_press(Message::CopyToClipboard(self.response.clone())),
            ]
            .spacing(10)
        )
        .height(Length::Fill);

        if self.is_generating {
            column![
                text("Generating response..."),
                response_view,
            ]
            .spacing(10)
            .into()
        } else {
            response_view.into()
        }
    }

    fn view_settings(&self) -> Element<Message> {
        let model_status = if self.model_loaded {
            text("Model loaded").style(iced::Color::from([0.0, 1.0, 0.0]))
        } else {
            text("No model loaded").style(iced::Color::from([1.0, 0.0, 0.0]))
        };

        column![
            row![
                text("Model Path:"),
                text_input("path/to/model.gguf", &self.model_path)
                    .on_input(Message::ModelPathChanged)
                    .width(400),
                button("Load").on_press(Message::LoadModel),
            ]
            .spacing(10),
            model_status,
        ]
        .spacing(20)
        .into()
    }

    fn generate_with_prompt(&mut self, prompt_prefix: &str) -> Command<Message> {
        if !self.model_loaded || self.is_generating {
            return Command::none();
        }

        self.is_generating = true;
        let code = self.code_content.text();
        let full_prompt = format!("{}{}\n\nLanguage: {}", prompt_prefix, code, self.language);
        let engine = self.engine.clone();

        Command::perform(
            async move {
                let mut engine_guard = engine.lock().await;
                if let Some(engine) = engine_guard.as_mut() {
                    let config = GenerationConfig {
                        max_tokens: 500,
                        temperature: 0.7,
                        top_p: 0.9,
                        ..Default::default()
                    };
                    
                    match engine.generate(&full_prompt, &config).await {
                        Ok(response) => response,
                        Err(e) => format!("Error generating response: {}", e),
                    }
                } else {
                    "No model loaded".to_string()
                }
            },
            Message::ResponseReceived,
        )
    }
}

// Helper module for syntax highlighting
mod syntax_highlight {
    use super::*;

    pub fn highlight_code(
        code: &str,
        language: &str,
        syntax_set: &SyntaxSet,
        theme_set: &ThemeSet,
    ) -> Vec<(Style, String)> {
        let syntax = syntax_set
            .find_syntax_by_extension(language)
            .unwrap_or_else(|| syntax_set.find_syntax_plain_text());
        
        let mut highlighter = HighlightLines::new(syntax, &theme_set.themes["base16-ocean.dark"]);
        let mut highlighted = Vec::new();

        for line in LinesWithEndings::from(code) {
            let ranges = highlighter.highlight_line(line, &syntax_set).unwrap();
            for (style, text) in ranges {
                highlighted.push((style, text.to_string()));
            }
        }

        highlighted
    }
}