use std::process::{Command, Output, ExitStatus};
use std::io;

pub struct OmegaProcess {
    command: String,
    args: Vec<String>,
    working_dir: Option<String>,
    env: Vec<(String, String)>,
    stdin_data: Option<Vec<u8>>,
}

impl OmegaProcess {
    pub fn new(command: &str) -> Self {
        Self {
            command: command.to_string(),
            args: Vec::new(),
            working_dir: None,
            env: Vec::new(),
            stdin_data: None,
        }
    }

    pub fn arg(mut self, arg: &str) -> Self {
        self.args.push(arg.to_string());
        self
    }

    pub fn args(mut self, args: &[&str]) -> Self {
        self.args.extend(args.iter().map(|s| s.to_string()));
        self
    }

    pub fn working_dir(mut self, dir: &str) -> Self {
        self.working_dir = Some(dir.to_string());
        self
    }

    pub fn env(mut self, key: &str, value: &str) -> Self {
        self.env.push((key.to_string(), value.to_string()));
        self
    }

    pub fn stdin(mut self, data: &[u8]) -> Self {
        self.stdin_data = Some(data.to_vec());
        self
    }

    pub fn stdin_string(mut self, data: &str) -> Self {
        self.stdin_data = Some(data.as_bytes().to_vec());
        self
    }

    pub fn run(&self) -> io::Result<ProcessResult> {
        let mut cmd = Command::new(&self.command);
        cmd.args(&self.args);

        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        for (key, value) in &self.env {
            cmd.env(key, value);
        }

        let output = if self.stdin_data.is_some() {
            let mut child = cmd.stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()?;

            if let Some(ref data) = self.stdin_data {
                use std::io::Write;
                child.stdin.as_mut().unwrap().write_all(data)?;
            }

            child.wait_with_output()?
        } else {
            cmd.output()?
        };

        Ok(ProcessResult {
            status: output.status,
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }

    pub fn run_string(&self) -> io::Result<String> {
        let result = self.run()?;
        Ok(String::from_utf8_lossy(&result.stdout).to_string())
    }

    pub fn run_status(&self) -> io::Result<i32> {
        let result = self.run()?;
        Ok(result.status.code().unwrap_or(1))
    }

    pub fn run_background(&self) -> io::Result<BackgroundProcess> {
        let mut cmd = Command::new(&self.command);
        cmd.args(&self.args);

        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        for (key, value) in &self.env {
            cmd.env(key, value);
        }

        let child = cmd.spawn()?;
        Ok(BackgroundProcess { child })
    }
}

pub struct ProcessResult {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl ProcessResult {
    pub fn success(&self) -> bool {
        self.status.success()
    }

    pub fn exit_code(&self) -> i32 {
        self.status.code().unwrap_or(1)
    }

    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }

    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }

    pub fn stdout_string(&self) -> String {
        String::from_utf8_lossy(&self.stdout).to_string()
    }

    pub fn stderr_string(&self) -> String {
        String::from_utf8_lossy(&self.stderr).to_string()
    }
}

pub struct BackgroundProcess {
    child: std::process::Child,
}

impl BackgroundProcess {
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        self.child.wait()
    }

    pub fn kill(&mut self) -> io::Result<()> {
        self.child.kill()
    }

    pub fn is_running(&mut self) -> bool {
        self.child.try_wait().ok().flatten().is_none()
    }
}

// Shell command execution
pub fn shell(command: &str) -> io::Result<ProcessResult> {
    if cfg!(target_os = "windows") {
        OmegaProcess::new("cmd")
            .args(&["/C", command])
            .run()
    } else {
        OmegaProcess::new("sh")
            .args(&["-c", command])
            .run()
    }
}

pub fn shell_string(command: &str) -> io::Result<String> {
    shell(command).map(|r| r.stdout_string())
}

pub fn shell_status(command: &str) -> io::Result<i32> {
    shell(command).map(|r| r.exit_code())
}

// Environment variables
pub fn get_env(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

pub fn set_env(key: &str, value: &str) {
    std::env::set_var(key, value);
}

pub fn remove_env(key: &str) {
    std::env::remove_var(key);
}

pub fn env_vars() -> Vec<(String, String)> {
    std::env::vars().collect()
}

// Process information
pub fn pid() -> u32 {
    std::process::id()
}

pub fn exit(code: i32) -> ! {
    std::process::exit(code)
}

pub fn abort() -> ! {
    std::process::abort()
}
