import subprocess, os, shutil

VALIDATORS = '''"""Validation helpers for widgetkit."""
import re


def validate_email(address):
    if not address or "@" not in address:
        return False
    local, _, domain = address.partition("@")
    if not local or not domain or "." not in domain:
        return False
    return bool(re.match(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", address))


def validate_phone(number):
    digits = re.sub(r"\\D", "", number)
    return len(digits) in (10, 11)


def validate_zip(code):
    return bool(re.match(r"^\\d{5}(-\\d{4})?$", code))


def validate_age(age, minimum=0, maximum=150):
    return minimum <= age <= maximum


def validate_password_strength(password):
    if len(password) < 8:
        return False
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    return has_upper and has_lower and has_digit
'''

FORMATTING = '''"""Formatting helpers for widgetkit."""


def format_currency(amount, currency="USD"):
    symbols = {"USD": "$", "EUR": "\\u20ac", "GBP": "\\u00a3"}
    symbol = symbols.get(currency, currency + " ")
    return f"{symbol}{amount:,.2f}"


def format_date(year, month, day):
    return f"{year:04d}-{month:02d}-{day:02d}"


def format_name(first, last, middle=None):
    if middle:
        return f"{first} {middle[0]}. {last}"
    return f"{first} {last}"


def truncate(text, length=80, suffix="..."):
    if len(text) <= length:
        return text
    return text[: length - len(suffix)] + suffix


def pluralize(count, singular, plural=None):
    if count == 1:
        return f"{count} {singular}"
    return f"{count} {plural or singular + 's'}"
'''

INVENTORY = '''"""Simple in-memory inventory tracker."""


class Inventory:
    def __init__(self, low_stock_threshold=5):
        self.items = {}
        self.low_stock_threshold = low_stock_threshold

    def add_item(self, sku, quantity, unit_price):
        if sku in self.items:
            self.items[sku]["quantity"] += quantity
        else:
            self.items[sku] = {"quantity": quantity, "unit_price": unit_price}

    def remove_item(self, sku, quantity):
        if sku not in self.items:
            raise KeyError(f"unknown sku: {sku}")
        if self.items[sku]["quantity"] < quantity:
            raise ValueError("not enough stock")
        self.items[sku]["quantity"] -= quantity

    def get_quantity(self, sku):
        return self.items.get(sku, {}).get("quantity", 0)

    def low_stock_items(self):
        return [sku for sku, info in self.items.items() if info["quantity"] <= self.low_stock_threshold]

    def total_value(self):
        return sum(info["quantity"] * info["unit_price"] for info in self.items.values())
'''

ORDERS = '''"""Order processing pipeline."""
from .inventory import Inventory


def compute_tax(subtotal, rate=0.08):
    return round(subtotal * rate, 2)


class OrderProcessor:
    def __init__(self, inventory, discount_codes=None):
        self.inventory = inventory
        self.discount_codes = discount_codes or {}
        self.log = []

    def validate_order(self, items):
        for sku, quantity in items:
            if self.inventory.get_quantity(sku) < quantity:
                return False
        return True

    def apply_discount(self, subtotal, code):
        pct = self.discount_codes.get(code, 0)
        return round(subtotal * (1 - pct), 2)

    def process(self, items, discount_code=None):
        if not self.validate_order(items):
            self._log("order rejected: insufficient stock")
            return None
        subtotal = sum(self.inventory.items[sku]["unit_price"] * qty for sku, qty in items)
        if discount_code:
            subtotal = self.apply_discount(subtotal, discount_code)
        tax = compute_tax(subtotal)
        for sku, qty in items:
            self.inventory.remove_item(sku, qty)
        total = subtotal + tax
        self._log(f"order processed: total={total}")
        return total

    def _log(self, message):
        self.log.append(message)
'''

UTILS = '''"""Misc helpers."""


def chunk_list(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result


def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def unique_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def safe_divide(numerator, denominator, default=0.0):
    if denominator == 0:
        return default
    return numerator / denominator
'''

REPORTS = '''"""Reporting helpers."""
from .formatting import format_currency


def generate_summary(orders):
    total = sum(orders)
    count = len(orders)
    average = total / count if count else 0
    return {"total": total, "count": count, "average": average}


def top_customers(customer_totals, n=3):
    ranked = sorted(customer_totals.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n]


def format_report(summary):
    lines = [
        f"Total: {format_currency(summary['total'])}",
        f"Orders: {summary['count']}",
        f"Average: {format_currency(summary['average'])}",
    ]
    return "\\n".join(lines)
'''

INIT = '"""widgetkit: a small internal utility library."""\n'

README = "# widgetkit\n\nInternal utility library. No type checking is currently enforced.\n"

FILES = {
    "src/widgetkit/__init__.py": INIT,
    "src/widgetkit/validators.py": VALIDATORS,
    "src/widgetkit/formatting.py": FORMATTING,
    "src/widgetkit/inventory.py": INVENTORY,
    "src/widgetkit/orders.py": ORDERS,
    "src/widgetkit/utils.py": UTILS,
    "src/widgetkit/reports.py": REPORTS,
    "README.md": README,
}


def materialize_repo(dest_dir: str) -> str:
    """Fresh copy of the widgetkit template repo, git-initialized with one
    baseline commit (representing 'existing untyped legacy code')."""
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    for relpath, content in FILES.items():
        full = os.path.join(dest_dir, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
    env = os.environ.copy()
    subprocess.run(["git", "init", "-q"], cwd=dest_dir, check=True, env=env)
    subprocess.run(["git", "config", "user.email", "dev@widgetkit.local"], cwd=dest_dir, check=True, env=env)
    subprocess.run(["git", "config", "user.name", "WidgetKit Dev"], cwd=dest_dir, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=dest_dir, check=True, env=env)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Initial commit: widgetkit v0.1 (no type annotations yet)"],
        cwd=dest_dir, check=True, env=env,
    )
    return dest_dir


if __name__ == "__main__":
    import sys
    d = materialize_repo("/tmp/widgetkit_probe")
    print("materialized at", d)
