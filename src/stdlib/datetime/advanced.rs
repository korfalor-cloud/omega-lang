//! Advanced datetime operations: calendar, timezone, recurrence, and business day logic.

use super::datetime::{OmegaDateTime, OmegaDuration};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Calendar helpers
// ---------------------------------------------------------------------------

/// Returns the number of days in a given year.
pub fn days_in_year(year: i32) -> u16 {
    if OmegaDateTime::is_leap_year(year) {
        366
    } else {
        365
    }
}

/// Returns the ordinal day-of-year (1-based) for the given date.
pub fn ordinal_day(year: i32, month: u8, day: u8) -> u16 {
    let mut ord = day as u16;
    for m in 1..month {
        ord += OmegaDateTime::days_in_month(year, m) as u16;
    }
    ord
}

/// Computes the ISO week-number and ISO week-year for a given Gregorian date.
/// Returns (iso_week_year, iso_week_number, iso_weekday 1=Mon..7=Sun).
pub fn iso_week_number(year: i32, month: u8, day: u8) -> (i32, u8, u8) {
    // Use the timestamp-based approach for correctness.
    let dt = OmegaDateTime::new(year, month, day, 0, 0, 0);
    let ts = dt.to_timestamp();
    let days_since_epoch = ts / 86400; // days since 1970-01-01 (which was Thursday)

    // ISO weekday: 1=Monday .. 7=Sunday
    let iso_wd = ((days_since_epoch % 7 + 4) % 7) as u8;
    let iso_wd = if iso_wd == 0 { 7 } else { iso_wd };

    // Thursday of this ISO week determines the ISO year.
    let thursday_ts = ts + (4 - iso_wd as i64) * 86400;
    let thurs = OmegaDateTime::from_timestamp(thursday_ts);

    // ISO week number: ordinal of Thursday divided by 7, plus 1.
    let thurs_ordinal = ordinal_day(thurs.year, thurs.month, thurs.day);
    let week = ((thurs_ordinal - 1) / 7) + 1;

    (thurs.year, week as u8, iso_wd)
}

/// Returns the first day-of-week (0=Sun .. 6=Sat) on or after the given date.
/// Useful for computing "first Monday of month", etc.
pub fn first_dow_on_or_after(year: i32, month: u8, day: u8, target_dow: u8) -> u8 {
    let dt = OmegaDateTime::new(year, month, day, 0, 0, 0);
    let current = dt.day_of_week();
    let diff = (target_dow + 7 - current) % 7;
    day + diff
}

// ---------------------------------------------------------------------------
// Timezone handling
// ---------------------------------------------------------------------------

/// Supported named time zones with their UTC offset in minutes and optional
/// daylight-saving abbreviation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TzInfo {
    pub name: &'static str,
    pub offset_minutes: i16,
    pub dst_offset_minutes: i16, // additional offset when DST is active
}

/// A small static timezone table covering the most common zones.
pub const TZ_TABLE: &[TzInfo] = &[
    TzInfo { name: "UTC", offset_minutes: 0, dst_offset_minutes: 0 },
    TzInfo { name: "EST", offset_minutes: -300, dst_offset_minutes: 60 },
    TzInfo { name: "CST", offset_minutes: -360, dst_offset_minutes: 60 },
    TzInfo { name: "MST", offset_minutes: -420, dst_offset_minutes: 60 },
    TzInfo { name: "PST", offset_minutes: -480, dst_offset_minutes: 60 },
    TzInfo { name: "CET", offset_minutes: 60, dst_offset_minutes: 60 },
    TzInfo { name: "EET", offset_minutes: 120, dst_offset_minutes: 60 },
    TzInfo { name: "IST", offset_minutes: 330, dst_offset_minutes: 0 },
    TzInfo { name: "JST", offset_minutes: 540, dst_offset_minutes: 0 },
    TzInfo { name: "AEST", offset_minutes: 600, dst_offset_minutes: 60 },
];

/// Looks up a timezone by abbreviation. Returns `None` if not found.
pub fn find_tz(name: &str) -> Option<&'static TzInfo> {
    TZ_TABLE.iter().find(|t| t.name.eq_ignore_ascii_case(name))
}

/// Converts an `OmegaDateTime` from one timezone to another (offsets in minutes).
pub fn convert_timezone(dt: &OmegaDateTime, from_offset: i16, to_offset: i16) -> OmegaDateTime {
    let diff = (to_offset - from_offset) as i64;
    let mut out = dt.clone();
    out.add_minutes(diff);
    out.timezone_offset = to_offset;
    out
}

/// Returns true if the given date falls inside US daylight-saving time
/// (second Sunday in March, 02:00 -> first Sunday in November, 02:00).
pub fn is_us_dst(year: i32, month: u8, day: u8, hour: u8) -> bool {
    if month < 3 || month > 11 {
        return false;
    }
    if month > 3 && month < 11 {
        return true;
    }
    // March
    if month == 3 {
        let dst_start = first_dow_on_or_after(year, 3, 1, 0) + 7; // second Sunday
        if day > dst_start {
            return true;
        }
        if day == dst_start && hour >= 2 {
            return true;
        }
        return false;
    }
    // November
    let dst_end = first_dow_on_or_after(year, 11, 1, 0); // first Sunday
    if day < dst_end {
        return true;
    }
    if day == dst_end && hour < 2 {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Duration arithmetic
// ---------------------------------------------------------------------------

/// Adds a number of whole months to a date, clamping the day to the last valid
/// day of the target month.
pub fn add_months_clamped(year: i32, month: u8, day: u8, months: i32) -> (i32, u8, u8) {
    let total = (year * 12 + month as i32 - 1) + months;
    let y = total / 12;
    let m = (total % 12 + 1) as u8;
    let max_day = OmegaDateTime::days_in_month(y, m);
    let d = if day > max_day { max_day } else { day };
    (y, m, d)
}

/// Adds a number of whole years, clamping Feb-29 -> Feb-28 if necessary.
pub fn add_years_clamped(year: i32, month: u8, day: u8, years: i32) -> (i32, u8, u8) {
    let y = year + years;
    let d = if month == 2 && day == 29 && !OmegaDateTime::is_leap_year(y) {
        28
    } else {
        day
    };
    (y, month, d)
}

/// Returns the precise duration between two datetimes in seconds (signed).
pub fn duration_between(a: &OmegaDateTime, b: &OmegaDateTime) -> OmegaDuration {
    let diff = b.to_timestamp() - a.to_timestamp();
    OmegaDuration::from_seconds(diff)
}

/// Computes the "calendar difference" between two dates: years, months, days.
/// Useful for age calculations.
pub fn calendar_diff(
    y1: i32, m1: u8, d1: u8,
    y2: i32, m2: u8, d2: u8,
) -> (i32, u8, u8) {
    let mut years = y2 - y1;
    let mut months = m2 as i32 - m1 as i32;
    let mut days = d2 as i32 - d1 as i32;

    if days < 0 {
        months -= 1;
        // Borrow days from the previous month of the end date.
        let (pm_y, pm_m) = if m2 == 1 {
            (y2 - 1, 12)
        } else {
            (y2, m2 - 1)
        };
        days += OmegaDateTime::days_in_month(pm_y, pm_m) as i32;
    }
    if months < 0 {
        years -= 1;
        months += 12;
    }

    (years, months as u8, days as u8)
}

// ---------------------------------------------------------------------------
// Recurrence rules (RFC 5545 subset)
// ---------------------------------------------------------------------------

/// Frequency for recurrence expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RRuleFreq {
    Secondly,
    Minutely,
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

/// A simplified RFC 5545 RRULE.
#[derive(Debug, Clone)]
pub struct RRule {
    pub freq: RRuleFreq,
    pub interval: u32,           // default 1
    pub count: Option<u32>,      // mutually exclusive with until
    pub until: Option<OmegaDateTime>,
    pub by_day: Option<Vec<u8>>, // 0=Sun..6=Sat
    pub by_month: Option<Vec<u8>>,
    pub by_month_day: Option<Vec<i8>>, // -31..31 (negative = from end)
    pub by_set_pos: Option<Vec<i32>>,
    pub wkst: u8,                // week start day (0=Sun)
}

impl RRule {
    /// Creates a simple daily recurrence.
    pub fn daily(interval: u32) -> Self {
        Self {
            freq: RRuleFreq::Daily,
            interval,
            count: None,
            until: None,
            by_day: None,
            by_month: None,
            by_month_day: None,
            by_set_pos: None,
            wkst: 0,
        }
    }

    /// Creates a simple weekly recurrence with optional days-of-week.
    pub fn weekly(interval: u32, by_day: Option<Vec<u8>>) -> Self {
        Self {
            freq: RRuleFreq::Weekly,
            interval,
            count: None,
            until: None,
            by_day,
            by_month: None,
            by_month_day: None,
            by_set_pos: None,
            wkst: 0,
        }
    }

    /// Creates a simple monthly recurrence.
    pub fn monthly(interval: u32) -> Self {
        Self {
            freq: RRuleFreq::Monthly,
            interval,
            count: None,
            until: None,
            by_day: None,
            by_month: None,
            by_month_day: None,
            by_set_pos: None,
            wkst: 0,
        }
    }

    /// Creates a simple yearly recurrence.
    pub fn yearly(interval: u32) -> Self {
        Self {
            freq: RRuleFreq::Yearly,
            interval,
            count: None,
            until: None,
            by_day: None,
            by_month: None,
            by_month_day: None,
            by_set_pos: None,
            wkst: 0,
        }
    }

    /// Builder: set count limit.
    pub fn with_count(mut self, n: u32) -> Self {
        self.count = Some(n);
        self.until = None;
        self
    }

    /// Builder: set until date.
    pub fn with_until(mut self, dt: OmegaDateTime) -> Self {
        self.until = Some(dt);
        self.count = None;
        self
    }
}

/// Expand a recurrence rule starting from `dtstart`, returning up to `max`
/// occurrences.
pub fn rrule_expand(dtstart: &OmegaDateTime, rule: &RRule, max: usize) -> Vec<OmegaDateTime> {
    let mut results = Vec::new();
    let mut current = dtstart.clone();
    let mut generated: u32 = 0;
    let limit = rule.count.unwrap_or(max as u32);

    // Safety cap to avoid infinite loops.
    let max_iterations = max.min(10_000);

    for _ in 0..max_iterations {
        if results.len() >= max {
            break;
        }
        if let Some(ref until) = rule.until {
            if current.to_timestamp() > until.to_timestamp() {
                break;
            }
        }
        if generated >= limit {
            break;
        }

        if matches_by_rules(&current, rule) {
            results.push(current.clone());
            generated += 1;
        }

        advance_candidate(&mut current, rule);
    }

    results
}

fn advance_candidate(dt: &mut OmegaDateTime, rule: &RRule) {
    match rule.freq {
        RRuleFreq::Secondly => dt.add_seconds(rule.interval as i64),
        RRuleFreq::Minutely => dt.add_minutes(rule.interval as i64),
        RRuleFreq::Hourly => dt.add_hours(rule.interval as i64),
        RRuleFreq::Daily => dt.add_days(rule.interval as i64),
        RRuleFreq::Weekly => dt.add_days(7 * rule.interval as i64),
        RRuleFreq::Monthly => dt.add_months(rule.interval as i32),
        RRuleFreq::Yearly => dt.add_years(rule.interval as i32),
    }
}

fn matches_by_rules(dt: &OmegaDateTime, rule: &RRule) -> bool {
    if let Some(ref days) = rule.by_day {
        if !days.contains(&dt.day_of_week()) {
            return false;
        }
    }
    if let Some(ref months) = rule.by_month {
        if !months.contains(&dt.month) {
            return false;
        }
    }
    if let Some(ref mdays) = rule.by_month_day {
        let day_i = dt.day as i8;
        let days_in = OmegaDateTime::days_in_month(dt.year, dt.month) as i8;
        let neg_day = day_i - days_in - 1; // e.g. last day = -1
        if !mdays.contains(&day_i) && !mdays.contains(&neg_day) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Business day calculations
// ---------------------------------------------------------------------------

/// Set of holiday dates represented as (month, day) for fixed holidays and a
/// callback for computed holidays.
#[derive(Debug, Clone)]
pub struct BusinessCalendar {
    /// Fixed holidays as (month, day).
    pub fixed_holidays: HashSet<(u8, u8)>,
    /// Observed holiday dates as timestamps (for computed holidays).
    pub observed_holidays: HashSet<i64>,
    /// Which days are considered weekends (bit flags: 0=Sun..6=Sat).
    pub weekend_mask: u8,
}

impl BusinessCalendar {
    /// Creates a US-style business calendar (Sat/Sun weekend, major US holidays).
    pub fn us(year: i32) -> Self {
        let mut cal = Self {
            fixed_holidays: HashSet::new(),
            observed_holidays: HashSet::new(),
            weekend_mask: 0b0100_0001, // Sun + Sat
        };
        cal.add_fixed(1, 1);   // New Year's Day
        cal.add_fixed(7, 4);   // Independence Day
        cal.add_fixed(12, 25); // Christmas
        // MLK Day: 3rd Monday of January
        cal.add_nth_weekday(year, 1, 1, 3);
        // Presidents' Day: 3rd Monday of February
        cal.add_nth_weekday(year, 2, 1, 3);
        // Memorial Day: last Monday of May
        cal.add_last_weekday(year, 5, 1);
        // Labor Day: 1st Monday of September
        cal.add_nth_weekday(year, 9, 1, 1);
        // Thanksgiving: 4th Thursday of November
        cal.add_nth_weekday(year, 11, 4, 4);
        cal
    }

    pub fn add_fixed(&mut self, month: u8, day: u8) {
        self.fixed_holidays.insert((month, day));
    }

    /// Adds the nth occurrence of a weekday in a given month (0=Sun..6=Sat).
    pub fn add_nth_weekday(&mut self, year: i32, month: u8, dow: u8, n: u8) {
        let first = first_dow_on_or_after(year, month, 1, dow);
        let target_day = first + 7 * (n - 1);
        let dt = OmegaDateTime::new(year, month, target_day, 0, 0, 0);
        self.observed_holidays.insert(dt.to_timestamp());
    }

    /// Adds the last occurrence of a weekday in a given month.
    pub fn add_last_weekday(&mut self, year: i32, month: u8, dow: u8) {
        let days_in = OmegaDateTime::days_in_month(year, month);
        let last_dt = OmegaDateTime::new(year, month, days_in, 0, 0, 0);
        let last_dow = last_dt.day_of_week();
        let diff = (last_dow + 7 - dow) % 7;
        let target_day = days_in - diff;
        let dt = OmegaDateTime::new(year, month, target_day, 0, 0, 0);
        self.observed_holidays.insert(dt.to_timestamp());
    }

    /// Returns true if the given date is a business day.
    pub fn is_business_day(&self, year: i32, month: u8, day: u8) -> bool {
        let dt = OmegaDateTime::new(year, month, day, 0, 0, 0);
        let dow = dt.day_of_week();

        // Weekend check
        if self.weekend_mask & (1 << dow) != 0 {
            return false;
        }

        // Fixed holiday
        if self.fixed_holidays.contains(&(month, day)) {
            return false;
        }

        // Observed holiday
        if self.observed_holidays.contains(&dt.to_timestamp()) {
            return false;
        }

        true
    }

    /// Returns the next business day on or after the given date.
    pub fn next_business_day(&self, year: i32, month: u8, day: u8) -> (i32, u8, u8) {
        let mut dt = OmegaDateTime::new(year, month, day, 0, 0, 0);
        loop {
            if self.is_business_day(dt.year, dt.month, dt.day) {
                return (dt.year, dt.month, dt.day);
            }
            dt.add_days(1);
        }
    }

    /// Returns the previous business day on or before the given date.
    pub fn prev_business_day(&self, year: i32, month: u8, day: u8) -> (i32, u8, u8) {
        let mut dt = OmegaDateTime::new(year, month, day, 0, 0, 0);
        loop {
            if self.is_business_day(dt.year, dt.month, dt.day) {
                return (dt.year, dt.month, dt.day);
            }
            dt.add_days(-1);
        }
    }

    /// Counts business days between two dates (exclusive of start, inclusive of
    /// end).
    pub fn business_days_between(
        &self,
        y1: i32, m1: u8, d1: u8,
        y2: i32, m2: u8, d2: u8,
    ) -> i32 {
        let ts1 = OmegaDateTime::new(y1, m1, d1, 0, 0, 0).to_timestamp();
        let ts2 = OmegaDateTime::new(y2, m2, d2, 0, 0, 0).to_timestamp();
        if ts2 <= ts1 {
            return 0;
        }
        let days = ((ts2 - ts1) / 86400) as i32;
        let mut count = 0;
        let mut dt = OmegaDateTime::new(y1, m1, d1, 0, 0, 0);
        for _ in 0..days {
            dt.add_days(1);
            if self.is_business_day(dt.year, dt.month, dt.day) {
                count += 1;
            }
        }
        count
    }

    /// Adds N business days to a date.
    pub fn add_business_days(
        &self,
        year: i32, month: u8, day: u8,
        n: i32,
    ) -> (i32, u8, u8) {
        let mut dt = OmegaDateTime::new(year, month, day, 0, 0, 0);
        let mut remaining = n.abs();
        let step: i64 = if n >= 0 { 1 } else { -1 };

        while remaining > 0 {
            dt.add_days(step);
            if self.is_business_day(dt.year, dt.month, dt.day) {
                remaining -= 1;
            }
        }
        (dt.year, dt.month, dt.day)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Calendar helpers ---

    #[test]
    fn test_days_in_year_leap() {
        assert_eq!(days_in_year(2000), 366);
        assert_eq!(days_in_year(1900), 365);
        assert_eq!(days_in_year(2024), 366);
        assert_eq!(days_in_year(2023), 365);
    }

    #[test]
    fn test_ordinal_day() {
        assert_eq!(ordinal_day(2024, 1, 1), 1);
        assert_eq!(ordinal_day(2024, 3, 1), 61); // leap year
        assert_eq!(ordinal_day(2023, 3, 1), 60); // non-leap
        assert_eq!(ordinal_day(2024, 12, 31), 366);
    }

    #[test]
    fn test_iso_week_number() {
        // 2024-01-01 is a Monday, ISO week 1 of 2024
        let (y, w, d) = iso_week_number(2024, 1, 1);
        assert_eq!(w, 1);
        assert_eq!(d, 1); // Monday

        // 2023-01-01 is Sunday => ISO week 52 of 2022
        let (y, w, _d) = iso_week_number(2023, 1, 1);
        assert_eq!(w, 52);
        assert_eq!(y, 2022);
    }

    #[test]
    fn test_first_dow_on_or_after() {
        // First Monday on or after Jan 1, 2024 (which is Monday itself)
        assert_eq!(first_dow_on_or_after(2024, 1, 1, 1), 1);
        // First Friday on or after Jan 1, 2024
        assert_eq!(first_dow_on_or_after(2024, 1, 1, 5), 5);
    }

    // --- Timezone ---

    #[test]
    fn test_find_tz() {
        let utc = find_tz("UTC").unwrap();
        assert_eq!(utc.offset_minutes, 0);

        let pst = find_tz("PST").unwrap();
        assert_eq!(pst.offset_minutes, -480);
        assert_eq!(pst.dst_offset_minutes, 60);

        assert!(find_tz("INVALID").is_none());
    }

    #[test]
    fn test_convert_timezone() {
        let dt = OmegaDateTime::new(2024, 6, 15, 12, 0, 0);
        let pst_dt = convert_timezone(&dt, 0, -480);
        assert_eq!(pst_dt.hour, 4);
        assert_eq!(pst_dt.timezone_offset, -480);
    }

    #[test]
    fn test_convert_timezone_wrap() {
        let dt = OmegaDateTime::new(2024, 6, 15, 2, 0, 0);
        let jst_dt = convert_timezone(&dt, 0, 540);
        assert_eq!(jst_dt.day, 15);
        assert_eq!(jst_dt.hour, 11);
    }

    #[test]
    fn test_is_us_dst() {
        // June is always DST
        assert!(is_us_dst(2024, 6, 15, 12));
        // January is never DST
        assert!(!is_us_dst(2024, 1, 15, 12));
        // Nov 3 2024 is first Sunday of November; 1am is still DST
        assert!(is_us_dst(2024, 11, 3, 1));
        // 2am is no longer DST
        assert!(!is_us_dst(2024, 11, 3, 2));
    }

    // --- Duration arithmetic ---

    #[test]
    fn test_add_months_clamped() {
        // Jan 31 + 1 month => Feb 29 (2024 leap) or Feb 28
        let (y, m, d) = add_months_clamped(2024, 1, 31, 1);
        assert_eq!((y, m, d), (2024, 2, 29));

        let (y, m, d) = add_months_clamped(2023, 1, 31, 1);
        assert_eq!((y, m, d), (2023, 2, 28));
    }

    #[test]
    fn test_add_years_clamped() {
        // Feb 29, 2024 + 1 year => Feb 28, 2025
        let (y, m, d) = add_years_clamped(2024, 2, 29, 1);
        assert_eq!((y, m, d), (2025, 2, 28));

        // Feb 29, 2024 + 4 years => Feb 29, 2028 (leap)
        let (y, m, d) = add_years_clamped(2024, 2, 29, 4);
        assert_eq!((y, m, d), (2028, 2, 29));
    }

    #[test]
    fn test_duration_between() {
        let a = OmegaDateTime::new(2024, 1, 1, 0, 0, 0);
        let b = OmegaDateTime::new(2024, 1, 2, 12, 0, 0);
        let dur = duration_between(&a, &b);
        assert_eq!(dur.as_seconds(), 86400 + 43200);
        assert_eq!(dur.as_hours(), 36.0);
    }

    #[test]
    fn test_calendar_diff() {
        // 2020-03-15 to 2024-07-20
        let (y, m, d) = calendar_diff(2020, 3, 15, 2024, 7, 20);
        assert_eq!(y, 4);
        assert_eq!(m, 4);
        assert_eq!(d, 5);
    }

    #[test]
    fn test_calendar_diff_same_date() {
        let (y, m, d) = calendar_diff(2024, 6, 15, 2024, 6, 15);
        assert_eq!((y, m, d), (0, 0, 0));
    }

    // --- Recurrence rules ---

    #[test]
    fn test_rrule_daily() {
        let start = OmegaDateTime::new(2024, 1, 1, 9, 0, 0);
        let rule = RRule::daily(1).with_count(5);
        let occurrences = rrule_expand(&start, &rule, 100);
        assert_eq!(occurrences.len(), 5);
        assert_eq!(occurrences[0].day, 1);
        assert_eq!(occurrences[4].day, 5);
    }

    #[test]
    fn test_rrule_weekly() {
        let start = OmegaDateTime::new(2024, 1, 1, 9, 0, 0); // Monday
        let rule = RRule::weekly(1, Some(vec![1, 5])).with_count(4); // Mon & Fri
        let occurrences = rrule_expand(&start, &rule, 100);
        assert_eq!(occurrences.len(), 4);
        assert_eq!(occurrences[0].day_of_week(), 1);
    }

    #[test]
    fn test_rrule_monthly() {
        let start = OmegaDateTime::new(2024, 1, 15, 0, 0, 0);
        let rule = RRule::monthly(1).with_count(3);
        let occurrences = rrule_expand(&start, &rule, 100);
        assert_eq!(occurrences.len(), 3);
        assert_eq!(occurrences[0].month, 1);
        assert_eq!(occurrences[1].month, 2);
        assert_eq!(occurrences[2].month, 3);
    }

    #[test]
    fn test_rrule_yearly() {
        let start = OmegaDateTime::new(2020, 6, 15, 0, 0, 0);
        let rule = RRule::yearly(1).with_count(4);
        let occurrences = rrule_expand(&start, &rule, 100);
        assert_eq!(occurrences.len(), 4);
        assert_eq!(occurrences[0].year, 2020);
        assert_eq!(occurrences[3].year, 2023);
    }

    #[test]
    fn test_rrule_with_until() {
        let start = OmegaDateTime::new(2024, 1, 1, 0, 0, 0);
        let until = OmegaDateTime::new(2024, 1, 5, 0, 0, 0);
        let rule = RRule::daily(1).with_until(until);
        let occurrences = rrule_expand(&start, &rule, 100);
        assert_eq!(occurrences.len(), 5); // 1st through 5th inclusive
    }

    #[test]
    fn test_rrule_by_month_filter() {
        let start = OmegaDateTime::new(2024, 1, 1, 0, 0, 0);
        let mut rule = RRule::monthly(1).with_count(12);
        rule.by_month = Some(vec![6, 7, 8]); // only summer months
        let occurrences = rrule_expand(&start, &rule, 100);
        assert_eq!(occurrences.len(), 3);
        for occ in &occurrences {
            assert!(occ.month >= 6 && occ.month <= 8);
        }
    }

    // --- Business day calculations ---

    #[test]
    fn test_business_calendar_us_weekend() {
        let cal = BusinessCalendar::us(2024);
        // Saturday
        assert!(!cal.is_business_day(2024, 6, 1));
        // Sunday
        assert!(!cal.is_business_day(2024, 6, 2));
        // Monday
        assert!(cal.is_business_day(2024, 6, 3));
    }

    #[test]
    fn test_business_calendar_us_fixed_holiday() {
        let cal = BusinessCalendar::us(2024);
        // July 4
        assert!(!cal.is_business_day(2024, 7, 4));
        // Dec 25
        assert!(!cal.is_business_day(2024, 12, 25));
    }

    #[test]
    fn test_business_calendar_us_computed_holidays() {
        let cal = BusinessCalendar::us(2024);
        // MLK Day: 3rd Monday of January 2024 = Jan 15
        assert!(!cal.is_business_day(2024, 1, 15));
        // Labor Day: 1st Monday of Sep 2024 = Sep 2
        assert!(!cal.is_business_day(2024, 9, 2));
        // Thanksgiving: 4th Thursday of Nov 2024 = Nov 28
        assert!(!cal.is_business_day(2024, 11, 28));
    }

    #[test]
    fn test_next_business_day() {
        let cal = BusinessCalendar::us(2024);
        // Friday June 14 -> Monday June 17
        let (y, m, d) = cal.next_business_day(2024, 6, 14);
        assert_eq!((y, m, d), (2024, 6, 17));
    }

    #[test]
    fn test_prev_business_day() {
        let cal = BusinessCalendar::us(2024);
        // Monday June 17 -> Friday June 14
        let (y, m, d) = cal.prev_business_day(2024, 6, 17);
        assert_eq!((y, m, d), (2024, 6, 14));
    }

    #[test]
    fn test_business_days_between() {
        let cal = BusinessCalendar::us(2024);
        // June 1 (Sat) to June 10 (Mon): 6 business days (Jun 3,4,5,6,7,10)
        let count = cal.business_days_between(2024, 6, 1, 2024, 6, 10);
        assert_eq!(count, 6);
    }

    #[test]
    fn test_add_business_days() {
        let cal = BusinessCalendar::us(2024);
        // Friday June 14 + 1 business day = Monday June 17
        let (y, m, d) = cal.add_business_days(2024, 6, 14, 1);
        assert_eq!((y, m, d), (2024, 6, 17));

        // Monday June 17 - 1 business day = Friday June 14
        let (y, m, d) = cal.add_business_days(2024, 6, 17, -1);
        assert_eq!((y, m, d), (2024, 6, 14));
    }

    #[test]
    fn test_add_business_days_span_weekend() {
        let cal = BusinessCalendar::us(2024);
        // Thursday June 13 + 2 = Monday June 17 (skips weekend)
        let (y, m, d) = cal.add_business_days(2024, 6, 13, 2);
        assert_eq!((y, m, d), (2024, 6, 17));
    }

    #[test]
    fn test_custom_business_calendar() {
        let mut cal = BusinessCalendar {
            fixed_holidays: HashSet::new(),
            observed_holidays: HashSet::new(),
            weekend_mask: 0b0100_0001, // Sat+Sun
        };
        cal.add_fixed(12, 25);
        // Dec 25 2024 is Wednesday => it's a holiday
        assert!(!cal.is_business_day(2024, 12, 25));
        // Dec 26 is a Thursday => regular business day
        assert!(cal.is_business_day(2024, 12, 26));
    }

}
