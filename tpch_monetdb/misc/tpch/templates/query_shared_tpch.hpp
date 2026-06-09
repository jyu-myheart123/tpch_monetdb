#pragma once

#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>

// Append a CSV field with minimal escaping for generated TPC-H query output.
inline void append_tpch_csv_field(std::string& out, const std::string& field) {
    const bool needs_quotes =
        field.find(',') != std::string::npos
        || field.find('"') != std::string::npos
        || field.find('\n') != std::string::npos
        || field.find('\r') != std::string::npos;
    if (!needs_quotes) {
        out.append(field);
        return;
    }
    out.push_back('"');
    for (const char ch : field) {
        if (ch == '"') {
            out.push_back('"');
        }
        out.push_back(ch);
    }
    out.push_back('"');
    return;
}

// Format numeric TPC-H values with stable fixed precision for validator parsing.
inline std::string format_tpch_double(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << value;
    return out.str();
}

// Convert a civil date to a day number using Howard Hinnant's algorithm.
inline int tpch_days_from_civil(int year, unsigned month, unsigned day) {
    year -= month <= 2;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(year - era * 400);
    const unsigned doy =
        (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int>(doe) - 719468;
}

// Convert a day number back to a civil date using Howard Hinnant's algorithm.
inline std::tuple<int, unsigned, unsigned> tpch_civil_from_days(int days) {
    days += 719468;
    const int era = (days >= 0 ? days : days - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(days - era * 146097);
    const unsigned yoe =
        (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int year = static_cast<int>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    const unsigned day = doy - (153 * mp + 2) / 5 + 1;
    const unsigned month = mp + (mp < 10 ? 3 : -9);
    year += month <= 2;
    return {year, month, day};
}

// Parse an ISO yyyy-mm-dd date to a comparable day number.
inline int tpch_parse_date_days(const std::string& date) {
    const int year = std::stoi(date.substr(0, 4));
    const unsigned month = static_cast<unsigned>(std::stoi(date.substr(5, 2)));
    const unsigned day = static_cast<unsigned>(std::stoi(date.substr(8, 2)));
    return tpch_days_from_civil(year, month, day);
}

// Format a civil date tuple as ISO yyyy-mm-dd.
inline std::string tpch_format_date(int year, unsigned month, unsigned day) {
    std::ostringstream out;
    out << std::setfill('0') << std::setw(4) << year
        << "-" << std::setw(2) << month
        << "-" << std::setw(2) << day;
    return out.str();
}

// Return whether one Gregorian year is a leap year.
inline bool tpch_is_leap_year(int year) {
    return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
}

// Return the number of days in one Gregorian month.
inline unsigned tpch_days_in_month(int year, unsigned month) {
    switch (month) {
        case 1:
        case 3:
        case 5:
        case 7:
        case 8:
        case 10:
        case 12:
            return 31;
        case 4:
        case 6:
        case 9:
        case 11:
            return 30;
        case 2:
            return tpch_is_leap_year(year) ? 29 : 28;
        default:
            return 31;
    }
}

// Add a number of days to an ISO yyyy-mm-dd date.
inline std::string tpch_date_add_days(const std::string& date, int delta_days) {
    const auto [year, month, day] =
        tpch_civil_from_days(tpch_parse_date_days(date) + delta_days);
    return tpch_format_date(year, month, day);
}

// Add a number of months to an ISO yyyy-mm-dd date.
inline std::string tpch_date_add_months(const std::string& date, int months) {
    int year = std::stoi(date.substr(0, 4));
    int month = std::stoi(date.substr(5, 2));
    unsigned day = static_cast<unsigned>(std::stoi(date.substr(8, 2)));
    month += months;
    while (month > 12) {
        month -= 12;
        ++year;
    }
    while (month < 1) {
        month += 12;
        --year;
    }
    const unsigned normalized_month = static_cast<unsigned>(month);
    const unsigned max_day = tpch_days_in_month(year, normalized_month);
    if (day > max_day) {
        day = max_day;
    }
    return tpch_format_date(year, normalized_month, day);
}

// Add a number of years to an ISO yyyy-mm-dd date.
inline std::string tpch_date_add_years(const std::string& date, int years) {
    const int year = std::stoi(date.substr(0, 4)) + years;
    const unsigned month = static_cast<unsigned>(std::stoi(date.substr(5, 2)));
    const unsigned day = static_cast<unsigned>(std::stoi(date.substr(8, 2)));
    return tpch_format_date(year, month, day);
}
