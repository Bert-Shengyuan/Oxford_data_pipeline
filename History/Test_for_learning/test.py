# range(1, 8) generates 1,2,3,4,5,6,7 -- it does NOT include 8

visits = 12

# TODO: if visits >= 20 -> tier = "Gold"
# TODO: elif visits >= 10 -> tier = "Silver"
# TODO: else -> tier = "Bronze"

if visits >= 20:
  tier = "Gold"
elif visits >= 10:
  tier = "Silver"
else:
  tier = "Bronze"

print("Customer tier: " + tier)


for day in range(1, 8):
    print("Business day " + str(day))

menu = {
    "Latte": 4.5,
    "Cappuccino": 4.0,
    "Americano": 3.5,
}

for aa, bb in menu.items():
    print(aa + " costs $" + str(bb))