"""Collect optional profile fields and generate a valid adult birth date."""

from datetime import date, timedelta
from secrets import choice, randbelow


SURNAMES = ("Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Phan", "Vũ", "Đặng")
GIVEN_NAMES = ("Mai", "Linh", "Thảo", "Vy", "Hân", "Ngọc", "Trang", "An")


def random_birth_date():
    start = date(1990, 1, 1)
    end = date(2003, 12, 31)
    return start + timedelta(days=randbelow((end - start).days + 1))


def prompt_profile(input_fn=input, output_fn=print):
    try:
        surname = input_fn("Họ (Enter để chọn ngẫu nhiên): ").strip()
        given_name = input_fn("Tên (Enter để chọn ngẫu nhiên): ").strip()
    except EOFError:
        surname = ""
        given_name = ""

    surname = surname or choice(SURNAMES)
    given_name = given_name or choice(GIVEN_NAMES)
    birth_date = random_birth_date()
    output_fn(f"Hồ sơ lần chạy này: {surname} {given_name}, sinh {birth_date:%d/%m/%Y}")
    return surname, given_name, birth_date
