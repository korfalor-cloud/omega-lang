use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OmegaDateTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub nanosecond: u32,
    pub timezone_offset: i16, // minutes from UTC
}

impl OmegaDateTime {
    pub fn new(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> Self {
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            nanosecond: 0,
            timezone_offset: 0,
        }
    }

    pub fn now() -> Self {
        // Simplified - would use system time in real implementation
        Self {
            year: 2024,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            nanosecond: 0,
            timezone_offset: 0,
        }
    }

    pub fn utc_now() -> Self {
        let mut dt = Self::now();
        dt.timezone_offset = 0;
        dt
    }

    pub fn from_timestamp(timestamp: i64) -> Self {
        let mut dt = Self::new(1970, 1, 1, 0, 0, 0);
        dt.add_seconds(timestamp);
        dt
    }

    pub fn to_timestamp(&self) -> i64 {
        let mut days: i64 = 0;

        // Days from years
        for y in 1970..self.year {
            days += if Self::is_leap_year(y) { 366 } else { 365 };
        }

        // Days from months
        for m in 1..self.month {
            days += Self::days_in_month(self.year, m) as i64;
        }

        // Days
        days += (self.day - 1) as i64;

        // Convert to seconds
        let seconds = days * 86400
            + self.hour as i64 * 3600
            + self.minute as i64 * 60
            + self.second as i64;

        seconds - self.timezone_offset as i64 * 60
    }

    pub fn is_leap_year(year: i32) -> bool {
        (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
    }

    pub fn days_in_month(year: i32, month: u8) -> u8 {
        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => if Self::is_leap_year(year) { 29 } else { 28 },
            _ => 0,
        }
    }

    pub fn day_of_week(&self) -> u8 {
        let ts = self.to_timestamp();
        let days = ts / 86400;
        ((days + 4) % 7) as u8 // 0 = Sunday
    }

    pub fn day_of_year(&self) -> u16 {
        let mut day = self.day as u16;
        for m in 1..self.month {
            day += Self::days_in_month(self.year, m) as u16;
        }
        day
    }

    pub fn week_of_year(&self) -> u8 {
        let doy = self.day_of_year();
        let dow = self.day_of_week();
        ((doy + 6 - dow as u16) / 7) as u8
    }

    pub fn add_years(&mut self, years: i32) {
        self.year += years;
        // Handle Feb 29 -> Feb 28
        if self.month == 2 && self.day == 29 && !Self::is_leap_year(self.year) {
            self.day = 28;
        }
    }

    pub fn add_months(&mut self, months: i32) {
        let total_months = (self.year * 12 + self.month as i32 - 1) + months;
        self.year = total_months / 12;
        self.month = (total_months % 12 + 1) as u8;

        let max_day = Self::days_in_month(self.year, self.month);
        if self.day > max_day {
            self.day = max_day;
        }
    }

    pub fn add_days(&mut self, days: i64) {
        let ts = self.to_timestamp() + days * 86400;
        let new_dt = Self::from_timestamp(ts);
        self.year = new_dt.year;
        self.month = new_dt.month;
        self.day = new_dt.day;
    }

    pub fn add_hours(&mut self, hours: i64) {
        self.add_seconds(hours * 3600);
    }

    pub fn add_minutes(&mut self, minutes: i64) {
        self.add_seconds(minutes * 60);
    }

    pub fn add_seconds(&mut self, seconds: i64) {
        let ts = self.to_timestamp() + seconds;
        let new_dt = Self::from_timestamp(ts);
        self.year = new_dt.year;
        self.month = new_dt.month;
        self.day = new_dt.day;
        self.hour = new_dt.hour;
        self.minute = new_dt.minute;
        self.second = new_dt.second;
    }

    pub fn difference_days(&self, other: &Self) -> i64 {
        let ts1 = self.to_timestamp();
        let ts2 = other.to_timestamp();
        (ts2 - ts1) / 86400
    }

    pub fn difference_seconds(&self, other: &Self) -> i64 {
        let ts1 = self.to_timestamp();
        let ts2 = other.to_timestamp();
        ts2 - ts1
    }

    pub fn format_iso8601(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day,
            self.hour, self.minute, self.second
        )
    }

    pub fn format_date(&self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }

    pub fn format_time(&self) -> String {
        format!("{:02}:{:02}:{:02}", self.hour, self.minute, self.second)
    }

    pub fn format_custom(&self, fmt: &str) -> String {
        let mut result = String::new();
        let mut chars = fmt.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '%' {
                match chars.next() {
                    Some('Y') => result.push_str(&format!("{:04}", self.year)),
                    Some('m') => result.push_str(&format!("{:02}", self.month)),
                    Some('d') => result.push_str(&format!("{:02}", self.day)),
                    Some('H') => result.push_str(&format!("{:02}", self.hour)),
                    Some('M') => result.push_str(&format!("{:02}", self.minute)),
                    Some('S') => result.push_str(&format!("{:02}", self.second)),
                    Some('y') => result.push_str(&format!("{:02}", self.year % 100)),
                    Some('j') => result.push_str(&format!("{:03}", self.day_of_year())),
                    Some('w') => result.push_str(&format!("{}", self.day_of_week())),
                    Some('W') => result.push_str(&format!("{:02}", self.week_of_year())),
                    Some('p') => {
                        if self.hour < 12 {
                            result.push_str("AM");
                        } else {
                            result.push_str("PM");
                        }
                    }
                    Some('I') => {
                        let h = if self.hour == 0 { 12 } else if self.hour > 12 { self.hour - 12 } else { self.hour };
                        result.push_str(&format!("{:02}", h));
                    }
                    Some('Z') => {
                        if self.timezone_offset >= 0 {
                            result.push_str(&format!("+{:02}:{:02}", self.timezone_offset / 60, self.timezone_offset % 60));
                        } else {
                            let offset = -self.timezone_offset;
                            result.push_str(&format!("-{:02}:{:02}", offset / 60, offset % 60));
                        }
                    }
                    Some('%') => result.push('%'),
                    Some(c) => {
                        result.push('%');
                        result.push(c);
                    }
                    None => result.push('%'),
                }
            } else {
                result.push(c);
            }
        }

        result
    }

    pub fn parse_iso8601(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split('T').collect();
        if parts.len() != 2 {
            return Err("Invalid ISO 8601 format".to_string());
        }

        let date_parts: Vec<&str> = parts[0].split('-').collect();
        if date_parts.len() != 3 {
            return Err("Invalid date format".to_string());
        }

        let time_parts: Vec<&str> = parts[1].split(':').collect();
        if time_parts.len() < 3 {
            return Err("Invalid time format".to_string());
        }

        Ok(Self::new(
            date_parts[0].parse().map_err(|_| "Invalid year")?,
            date_parts[1].parse().map_err(|_| "Invalid month")?,
            date_parts[2].parse().map_err(|_| "Invalid day")?,
            time_parts[0].parse().map_err(|_| "Invalid hour")?,
            time_parts[1].parse().map_err(|_| "Invalid minute")?,
            time_parts[2].parse().map_err(|_| "Invalid second")?,
        ))
    }

    pub fn weekday_name(&self) -> &str {
        match self.day_of_week() {
            0 => "Sunday",
            1 => "Monday",
            2 => "Tuesday",
            3 => "Wednesday",
            4 => "Thursday",
            5 => "Friday",
            6 => "Saturday",
            _ => "Unknown",
        }
    }

    pub fn month_name(&self) -> &str {
        match self.month {
            1 => "January",
            2 => "February",
            3 => "March",
            4 => "April",
            5 => "May",
            6 => "June",
            7 => "July",
            8 => "August",
            9 => "September",
            10 => "October",
            11 => "November",
            12 => "December",
            _ => "Unknown",
        }
    }

    pub fn is_valid(&self) -> bool {
        self.month >= 1
            && self.month <= 12
            && self.day >= 1
            && self.day <= Self::days_in_month(self.year, self.month)
            && self.hour < 24
            && self.minute < 60
            && self.second < 60
    }
}

impl fmt::Display for OmegaDateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            self.year, self.month, self.day,
            self.hour, self.minute, self.second
        )
    }
}

// Duration type
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OmegaDuration {
    pub seconds: i64,
    pub nanoseconds: i32,
}

impl OmegaDuration {
    pub fn new(seconds: i64, nanoseconds: i32) -> Self {
        Self { seconds, nanoseconds }
    }

    pub fn from_seconds(seconds: i64) -> Self {
        Self { seconds, nanoseconds: 0 }
    }

    pub fn from_minutes(minutes: i64) -> Self {
        Self { seconds: minutes * 60, nanoseconds: 0 }
    }

    pub fn from_hours(hours: i64) -> Self {
        Self { seconds: hours * 3600, nanoseconds: 0 }
    }

    pub fn from_days(days: i64) -> Self {
        Self { seconds: days * 86400, nanoseconds: 0 }
    }

    pub fn from_weeks(weeks: i64) -> Self {
        Self { seconds: weeks * 604800, nanoseconds: 0 }
    }

    pub fn as_seconds(&self) -> i64 {
        self.seconds
    }

    pub fn as_minutes(&self) -> f64 {
        self.seconds as f64 / 60.0
    }

    pub fn as_hours(&self) -> f64 {
        self.seconds as f64 / 3600.0
    }

    pub fn as_days(&self) -> f64 {
        self.seconds as f64 / 86400.0
    }

    pub fn as_weeks(&self) -> f64 {
        self.seconds as f64 / 604800.0
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut nanos = self.nanoseconds + other.nanoseconds;
        let mut secs = self.seconds + other.seconds;

        if nanos >= 1_000_000_000 {
            secs += 1;
            nanos -= 1_000_000_000;
        } else if nanos < 0 {
            secs -= 1;
            nanos += 1_000_000_000;
        }

        Self { seconds: secs, nanoseconds: nanos }
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut nanos = self.nanoseconds - other.nanoseconds;
        let mut secs = self.seconds - other.seconds;

        if nanos < 0 {
            secs -= 1;
            nanos += 1_000_000_000;
        }

        Self { seconds: secs, nanoseconds: nanos }
    }

    pub fn mul(&self, factor: i64) -> Self {
        let total_nanos = self.seconds as i128 * 1_000_000_000 + self.nanoseconds as i128;
        let result_nanos = total_nanos * factor as i128;

        Self {
            seconds: (result_nanos / 1_000_000_000) as i64,
            nanoseconds: (result_nanos % 1_000_000_000) as i32,
        }
    }

    pub fn abs(&self) -> Self {
        if self.seconds < 0 {
            Self { seconds: -self.seconds, nanoseconds: -self.nanoseconds }
        } else {
            self.clone()
        }
    }

    pub fn is_negative(&self) -> bool {
        self.seconds < 0
    }

    pub fn is_zero(&self) -> bool {
        self.seconds == 0 && self.nanoseconds == 0
    }
}

impl fmt::Display for OmegaDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_seconds = self.seconds.abs();
        let hours = total_seconds / 3600;
        let minutes = (total_seconds % 3600) / 60;
        let seconds = total_seconds % 60;

        if self.seconds < 0 {
            write!(f, "-")?;
        }

        if hours > 0 {
            write!(f, "{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            write!(f, "{}m {}s", minutes, seconds)
        } else {
            write!(f, "{}s", seconds)
        }
    }
}

// Timer utility
pub struct OmegaTimer {
    start: std::time::Instant,
    laps: Vec<std::time::Duration>,
}

impl OmegaTimer {
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
            laps: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.start = std::time::Instant::now();
        self.laps.clear();
    }

    pub fn lap(&mut self) -> std::time::Duration {
        let elapsed = self.start.elapsed();
        self.laps.push(elapsed);
        elapsed
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn laps(&self) -> &[std::time::Duration] {
        &self.laps
    }

    pub fn average_lap(&self) -> std::time::Duration {
        if self.laps.is_empty() {
            return std::time::Duration::ZERO;
        }
        let total: std::time::Duration = self.laps.iter().sum();
        total / self.laps.len() as u32
    }
}
