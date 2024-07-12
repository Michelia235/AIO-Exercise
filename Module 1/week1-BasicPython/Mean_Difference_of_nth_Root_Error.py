def MD_nRE(y, y_hat, n, p):
    y_root = y ** (1/ n )
    y_hat_root = y_hat ** (1/ n )
    difference = y_root - y_hat_root
    loss = difference ** p
    return loss
  

y = float(input("Predicted Value: "))
y_hat = float(input("Actual Value: "))
n = int(input("Placement of Root Error: "))
p = int(input("Power of loss: "))

MD_nRE(y, y_hat, n, p)