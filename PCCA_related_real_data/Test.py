def calculate_price_company_level(company_list):
  whole_price_list = []
  # print(whole_price_list)
  for index,good_label in enumerate(company_list):
    # good_price = calculate_total(price_list[index][0],price_list[index][1]) #等价
      good_price = calculate_total(good_label[0],good_label[1])
      whole_price_list.append(good_price)

  return whole_price_list

def calculate_total(price, quantity):
    """
    Calculate the total cost for a given price and quantity.
    """
    whole_price1 = price * quantity
    return whole_price1


A_price_list = [[1.2,3],[4.5,10],[3.4,15]]
B_price_list = [[9.1,17],[4.5,10],[3.99,15],[9.1,100]]


whole_price_list_A = calculate_price_company_level(A_price_list)

whole_price_list_B = calculate_price_company_level(B_price_list)