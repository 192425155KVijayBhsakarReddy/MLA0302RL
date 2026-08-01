ROWS = 4
COLS = 4

goal = (3,3)

policy = [["R"]*COLS for i in range(ROWS)]
V = [[0]*COLS for i in range(ROWS)]

gamma = 0.9

actions = {
    "U":(-1,0),
    "D":(1,0),
    "L":(0,-1),
    "R":(0,1)
}

for k in range(20):

    # Policy Evaluation

    for i in range(ROWS):
        for j in range(COLS):

            if (i,j)==goal:
                continue

            a=policy[i][j]

            dx,dy=actions[a]

            ni=i+dx
            nj=j+dy

            if ni<0 or ni>=ROWS or nj<0 or nj>=COLS:
                ni,nj=i,j

            V[i][j]=-1+gamma*V[ni][nj]

    # Policy Improvement

    stable=True

    for i in range(ROWS):
        for j in range(COLS):

            if (i,j)==goal:
                continue

            best=policy[i][j]
            bestvalue=-999

            for a,(dx,dy) in actions.items():

                ni=i+dx
                nj=j+dy

                if 0<=ni<ROWS and 0<=nj<COLS:

                    value=-1+gamma*V[ni][nj]

                    if value>bestvalue:
                        bestvalue=value
                        best=a

            if best!=policy[i][j]:
                stable=False

            policy[i][j]=best

    if stable:
        break

print("Optimal Policy")

for row in policy:
    print(row)
