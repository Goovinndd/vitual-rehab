def isHappy(n: int) -> bool:
    seen = set()
    while n not in seen:
        number = n
        seen.add(n)
        happy = 0
        while number > 0:
            happy = happy + (number%10)**2
            number = number//10
        if happy == 1:
            return True
        n = happy
    return False
if isHappy(19):
    print("1")
else:
    print("0")