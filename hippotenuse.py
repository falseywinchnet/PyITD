from mpmath import *
mp.dps = 500  # set precision

'''
             ..ed$$$$$$$$$$$$$be      *F..
       ^   z$$$$$$$$$$$$$$$$$$$$$$$$$$$$
          $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$.
         $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*e.
        4$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$be$$$$$$c
        4$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$L
        ^$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
         ^*$$$$$$$$$$$$$$$$$$$$$$$$$$$F^*$$$$$$$$$%
           ^"$$$$$$$$$$$$$$$$$$$$$$$$$    "**$P*"  Gilo94'
             4$$$F"3$$$$       4$$$$
             d$$$$ 4$$$$       4$$$$                 Hippo
                                                     ^^^^^
'''   
#with this function we provide the closed form algebraic expressions for deriving the hypotenuse of a triangle
#inscribing a unit circle of radius 1, where the triangle series begins by inscribing three points marked as the center,
#the point on the circle at 0, and the point on the circle at 90. That value can be determined empirically to be sqrt(2).
#for each value of n higher  than this we derive the identity of the triangle as a triangle with two sides of 1 and a rapidly 
#diminishing arc which divides in half every time.
#Going in the other direction, we may derive a value which could use the hypotenuse
#of the previous triangle and a rapidly diminishing but known a and b angle, culminating in n=2 with the last valid triangle.
#n=1 is for a triangle with zero area. n=0 is for a triangle that encapsulates a 360 degree circle...
#this is purely a hypothetical projection as we have no other criteria which may be used to determine their correct identity.
#however, we observe cosecant(pi/16) = 4 and cosecant(pi/32) = 5.cosecant(pi/8)=6, cosecant(pi/4)=1.
#other criteria that could be used is a further progession of the cosecant series but this quickly explodes to exponential values.
#the largest valid value of this projection series is (64 * (2 + sqrt(2 + sqrt(2 + sqrt(2))))**2 * (2 + sqrt(2 + sqrt(2 + sqrt(2 + sqrt(2)))))**2)/(2 - sqrt(2 + sqrt(2)))**2
#that is the hypotenuse, and the two sides are (4 *(2 + sqrt(2 + sqrt(2 + sqrt(2)))) *sqrt(4* (2 + sqrt(2 + sqrt(2 + sqrt(2 + sqrt(2)))))**2 + (2 + sqrt(2 + sqrt(2 + sqrt(2 + sqrt(2)))))**(5/2)))/(2 - sqrt(2 + sqrt(2)))
#the degrees are 0, 0, and 180.thereinlies the hippotenuse. i reject all alternatives as utterly unimaginative-
#the hippotenuse deserves extensive study for his unique and interesting qualities.
#alternatively we can use the simple cosine rule which is what i have done here.


mp.dps = 500  # set precision

def hypotenuse(x, n):
    value = 2
    if n < 0:
        for _ in range(abs(n)):
            value = 2 + sqrt(value)
        value = x * sqrt(value)
    elif n > 0:
        # First build up the nested expression
        for _ in range(abs(n)-1):
            value = 2 + sqrt(value)
        # Then apply the outer 2 - (nested expression)
        value = x * sqrt(2 - sqrt(value))
    else:
        value = x * sqrt(value)  #simplifies to sqrt(2) for 90 degrees
    return value

from mpmath import nstr

def get_hypotenuse_for_angle(a, angle):
    # Determine n based on the angle
    if angle == 90:
        return a * hypotenuse(1, 0)
    
    n = 1
    if angle < 90:
        while angle < 90 / pow(2, n):
            n += 1
        n_low, n_high = n - 1, n
        
    else:  # For angles greater than 90 (e.g., 135, 157.5, etc.)
        while 90- (angle -90 )< 90 / pow(2, n):
            n += 1
        n = -n
        n_low, n_high = n, n+1

    # Calculate the low and high angles based on n
    if n_low == 0: angle_low = 90
    elif n_low > 0: angle_low = 45 / (2 ** (n_low - 1))
    elif n_low == -1: angle_low = 135
    else: angle_low = 180 - (180 - 135) / (2 ** (-n_low - 1))

    if n_high == 0: angle_high = 90
    elif n_high > 0: angle_high = 45 / (2 ** (n_high - 1))
    elif n_high == -1: angle_high = 135
    else: angle_high = 180 - (180 - 135) / (2 ** (-n_high - 1))
    h_low = hypotenuse(a, n_low)
    h_high = hypotenuse(a, n_high)

    print("low : wider degree, maps to negative numbers, below zero below 90")
    print(n_low,angle_low,nstr(h_low))

    print("high: narrower degree, maps to positive numbers, above zero above 90")
    print(n_high,angle_high,nstr(h_high))
    result = h_low + ((angle - angle_low)/(angle_high-angle_low))* (h_high-h_low)
    return result

#note this method is not terribly accurate
