ROWS = 4
COLS = 4

goal = (3,3)

gamma = 0.9

V = [[0]*COLS for i in range(ROWS)]

actions = [(-1,0),(1,0),(0,-1),(0,1)]

while True:

    delta=0

    for i in range(ROWS):
        for j in range(COLS):

            if (i,j)==goal:
                continue

            best=-999

            for dx,dy in actions:

                ni=i+dx
                nj=j+dy

                if 0<=ni<ROWS and 0<=nj<COLS:

                    value=-1+gamma*V[ni][nj]

                    if value>best:
                        best=value

            delta=max(delta,abs(best-V[i][j]))

            V[i][j]=round(best,2)

    if delta<0.01:
        break

print("Value Function")

for row in V:
    print(row)

print("\nOptimal Policy")

symbols=["U","D","L","R"]

for i in range(ROWS):

    for j in range(COLS):

        if (i,j)==goal:
            print("G",end=" ")

        else:

            best=-999
            action=""

            for k,(dx,dy) in enumerate(actions):

                ni=i+dx
                nj=j+dy

                if 0<=ni<ROWS and 0<=nj<COLS:

                    value=-1+gamma*V[ni][nj]

                    if value>best:
                        best=value
                        action=symbols[k]

            print(action,end=" ")

    print()
