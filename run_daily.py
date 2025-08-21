
import json
import sys
from stock_analysis import analyze_stock, convert_to_builtin_type

# Optional: Only import email libraries if needed
def send_email(subject, body, to_addr, from_addr, password, smtp_server="smtp.gmail.com", smtp_port=587):
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg['Subject'] = subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(from_addr, password)
        server.sendmail(from_addr, to_addr, msg.as_string())

STOCKS_FILE = "stocks.txt"

def get_tickers(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]

def pretty_print_result(res):
    print("=" * 60)
    print(f"Ticker: {res.get('ticker', 'N/A')}")
    if "error" in res:
        print(f"ERROR: {res['error']}")
        return
    print(f"Current Price: ${res.get('current_price', 'N/A'):,.2f}")
    analyst = res.get("analyst_predictions", {})
    print(f"Analyst mean target: ${analyst.get('target_mean', 'N/A'):,.2f}")
    print(f"Analyst recommendation: {analyst.get('recommendation', 'N/A')}")
    print(f"Last inflection: {res.get('last_inflection', {})}")
    print(f"Volume analysis: {res.get('volume_analysis', {})}")
    action = res.get("inflection_action", {})
    print(f"Recommendation: {action.get('action', 'N/A')}")
    print(f"Analysis: {action.get('analysis', 'N/A')}")
    print("=" * 60)
    print()

def create_email_body(results):
    lines = []
    for res in results:
        if "error" in res:
            lines.append(f"{res['ticker']}: ERROR: {res['error']}")
            continue
        lines.append(f"Ticker: {res['ticker']}")
        lines.append(f"Current Price: {res.get('current_price', 'N/A'):,.2f}")
        analyst = res.get("analyst_predictions", {})
        lines.append(f"Analyst mean target: {analyst.get('target_mean', 'N/A'):,.2f}")
        lines.append(f"Analyst recommendation: {analyst.get('recommendation', 'N/A')}")
        lines.append(f"Last inflection: {res.get('last_inflection', {})}")
        lines.append(f"Volume analysis: {res.get('volume_analysis', {})}")
        action = res.get("inflection_action", {})
        lines.append(f"Recommendation: {action.get('action', 'N/A')}")
        lines.append(f"Analysis: {action.get('analysis', 'N/A')}")
        lines.append("-" * 40)
    return "\n".join(lines)

if __name__ == "__main__":
    # Usage: python run_daily.py terminal   OR   python run_daily.py email
    mode = sys.argv[1] if len(sys.argv) > 1 else "terminal"
    tickers = get_tickers(STOCKS_FILE)
    results = []
    for ticker in tickers:
        try:
            res = analyze_stock(ticker)
            res = convert_to_builtin_type(res)
        except Exception as e:
            res = {"ticker": ticker, "error": str(e)}
        results.append(res)

    if mode == "email":
        # Set these as you need (or use env vars for safety)
        EMAIL_TO = "norocelpopa@yahoo.com"
        EMAIL_FROM = "noro66@gmail.com"
        EMAIL_PASS = "ohuv fnkm tfcl jsoz"
        subject = f"Daily Stock Analysis "
        body = create_email_body(results)
        send_email(subject, body, EMAIL_TO, EMAIL_FROM, EMAIL_PASS)
        print(f"Email sent to {EMAIL_TO}!")
    else:  # default is terminal
        for res in results:
            pretty_print_result(res)