# Kaggle Santa 2023 - The Polytope Permutation Puzzle, KMCoders' Solution

https://www.kaggle.com/competitions/santa-2023/overview

Solving a 33x33x33 cube puzzle (281) in 4719 steps.

![0281.gif](0281.gif)

Solving a 9x50 globe puzzle (396) in 949 steps.

![0396.gif](0396.gif)

[Final submission](submission.csv)

[Scores](scores.csv)

# Solution

Among the three types of puzzles, the wreath puzzle was not particularly focused on because a very short solution could be found using simple beam search, and it had a lower total score compared to the others, making it a lower priority. Therefore, we concentrate on the solutions for the remaining two types, the cube and the globe puzzles. The strategies for solving both puzzles are based on the same principle.

Due to each move affecting a large number of pieces at once, simply applying moves as is can lead to a somewhat organized state, but fully solving the puzzle becomes extremely challenging. Thus, the key is to find sequences of moves that only swap a few pieces while leaving the rest unchanged. Remarkably, for both puzzles, there exist sequences of moves that only rotate three pieces and leave the rest unchanged, referred to as 3-rot.

- Example for the cube puzzle: d3.f2.d2.-f2.-d3.f2.-d2.-f2
- Example for the globe puzzle: f0.r0.f0.r1.f0.-r1.f0.-r0

When considering the cluster decomposition of all pieces (a cluster being a set of pieces that can be interchanged through moves), it turns out that for most clusters, except for special parts like the cube's corners, there exists a 3-rot for any three elements within them.

- For the cube puzzle: Corners, centers of faces, and centers of edges are exceptions. For other clusters, taking symmetry into account, it suffices to consider the diagonal parts of a 4x4x4 (1,1), the cross parts of a 5x5x5 (1,2), the non-diagonal and non-cross parts of a 6x6x6 (1,2), and the edge parts of a 4x4x4 (0,1). For each, a bidirectional search was performed to enumerate all shortest 3-rot for any three elements, with the longest being 14 moves.
- For the globe puzzle: The central row when the number of rows is odd is an exception. For other clusters, with the number of rows as 2 and the number of columns as 2c, R=f0.r0.f0.r1.f0.-r1.f0.-r0 corresponds to a 3-rot of ((0,0) (1,c) (1,c-1)). A 3-rot for any three pieces can be found by first finding a sequence of moves A that moves those three pieces to a state where R can be applied (where other pieces can move freely), using breadth-first search, and then the 3-rot can be obtained by A.R.-A.

When solving each cluster using 3-rot, it is important to note that all 3-rot are even permutations, so the overall permutation must also be even. Combining these insights, we can use the following approach:

1. Solve the special parts.
2. Operate without disrupting the solved parts so that all remaining clusters become even permutations.
3. Solve each cluster independently using 3-rot.

To achieve shorter solutions, we employ the following key ideas:

- Since 3-rot requires at least 8 moves and is lengthy, it's more efficient to bring the puzzle to a somewhat solved state using elementary moves or short sequences before employing 3-rot for the final touches.
- When adding a new sequence B to an existing sequence A, canceling out the end of A with the beginning of B can shorten the overall sequence. For example, if A=A'.ri and B=-ri.B', then A.B becomes A'.ri.-ri.B'=A'.B', thus saving 2 moves.
- If the current sequence is A=a[0]...a[T-1], instead of appending a 3-rot B for (i j k) at the end to form a[0]...a[T-1].B, at arbitrary time t, inserting a 3-rot B' for some (i' j' k') to achieve a[0]...a[t-1].B'.a[t]...a[T-1] results in the same state. Selecting the appropriate time t can make B' shorter than B or result in more cancellations. Therefore, rather than constructing the sequence from the front, it is better to try the insertions for all times and select the best time.

Based on these ideas, we use the following approach:

1. Solve the special parts.
2. Use elementary moves to make all clusters even while roughly aligning them.
3. Bring to a somewhat solved state using short sequences.
4. Insert 3-rots at arbitrary times.

## 1. Solve the special parts.

For the cube, the special clusters are 8 corners when n is even, and an additional 6 centers of each face and 12 centers of each edge when n is odd. For even n, these clusters can be solved by reducing them to a 2x2x2 cube problem, while for odd n, they can be solved by reducing to a 3x3x3 cube problem. The 2x2x2 cases were solved using bidirectional search, and the 3x3x3 cases were solved using an existing optimal solver available at [https://www.cflmath.com/Rubik/optimal_solver.html](https://www.cflmath.com/Rubik/optimal_solver.html).

In the case of the globe, the middle row when the number of rows is odd constitutes a special cluster. This row can be solved simply by rotating it left or right and does not change with the other moves, allowing it to be excluded from the problem.

## 2. Use elementary moves to make all clusters even while roughly aligning them (cube)

Face moves (such as r[0], r[n-1]) and center moves for odd n (such as r[(n−1)/2]) are not used as they would disrupt the parts solved in step 1. The remaining moves invert the parity of all affected clusters except for those that are diagonal. Therefore, by solving a system of linear equations in mod 2, it is possible to find a sequence of moves that makes all clusters even.

Starting with the sequence of moves determined in step 1, extended by adding the sequence found by solving the system of linear equations, we employed simulated annealing using the following moves as neighbors to keep each cluster even. The evaluation function used was "length of the move sequence + number of misaligned pieces × α (about 1.5)".

- Changing a single move. Moves like d[i].r[i] or d[i].r[n-1-i] do not alter the parity of any clusters, so d[i] can be changed to r[i] or r[n-1-i].
- Swapping the order of two adjacent moves. The swap from di.rj to rj.di corresponds to applying -rj.-di.rj.di, which actually corresponds to performing two sets of 3-rots, offering high locality and utility.
- Adding two moves. Pairs such as d[i] and r[i] or d[i] and r[n-1-i] can be inserted at any position.
- Removing two moves. Pairs such as d[i] and r[i] or d[i] and r[n-1-i] can be removed at once.

## 2. Use elementary moves to make all clusters even while roughly aligning them (globe)

For the globe, instead of trying to align each piece in its correct position, we adjust the positions by adding row shift moves at the end of the move sequence (e.g., performing two left shifts to transform the state from 450123 to 012345). In this case, the calculation of the number of aligned pieces in the intermediate step is to try all the offsets for each row and take the maximum value. However, when all pieces are of different colors, it's necessary to consider the parity. Performing a row shift move (ri) inverts the parity of the cluster that the row belongs to, so to achieve an even permutation after eliminating the offset, the "current permutation's parity" must match the "parity of the difference in offsets between the upper and lower rows." Therefore, in calculating the number of aligned pieces, the maximum number for even and odd offsets is computed for each row, and an appropriate combination from the upper and lower rows is selected.

In the case of the globe, each elementary move affects more pieces compared to the cube, making it difficult to achieve local changes. Thus, instead of simulated annealing, we employed a hill climbing with kicks.

The evaluation function is the sum of the following four values:
- The length of the current move sequence.
- The number of pieces in incorrect positions multiplied by 3.
- The number of pieces in the correct row but incorrect positions. Since an 8-move 3-rot moves pieces to the opposite row, pieces in the correct row but wrong positions are harder to align.
- The number of pairs with incorrect adjacency (e.g., 2 is right next to 0). Since every move preserves the adjacency of most pairs, it's beneficial when pieces are in the wrong position but maintain correct adjacency.

We used the following neighborhoods:
- Swapping two moves.
- Changing one move.
- Adding one move.
- Removing one move.

## 3. Bring to a somewhat solved state using short sequences (cube)

The 4-move sequence di.rj.-di.-rj corresponds to performing two 3-rots. This move is not only short, but also has high locality and is therefore useful. Furthermore, by extending this move to di.rj.rk.-di.-rj.-rk, it is possible to simultaneously execute two moves, di.rj.-di.-rj and di.rk.-di.-rk, within a shorter sequence.

Until the number of unsolved pieces falls below a certain proportion of the total (approximately 0.5), we employed a greedy method which finds a pair of (move sequence, insertion time) such that (increase in length of the sequence/increase in number of aligned pieces) is the smallest.

## 3. Bring to a somewhat solved state using short sequences (globe)

Unlike cube, we could not find short moves with high locality for the globe puzzle. Instead, our focus on the move sequence f0.r0.f0. This particular move effectively links and shifts the right half of the upper row with the left half of the lower row by one position, proving to be beneficial until a certain degree of alignment is achieved. Moreover, the globe puzzle presents a smaller number of clusters, each with a larger size compared to the cube, prompting us to devise a strategy where move sequences for each cluster (consisting of pairs of rows from the top and the bottom) are determined independently and subsequently merged.

In addressing each cluster, we employed three types of moves: the row shift (ri), the 3-move shift (fj.ri.fj), and the 3-rots, utilizing the same evaluation function as established in step 2. We conducted a beam search to maintain the top k sequences (approximately 100,000) with the lowest evaluation function values for each length. The sequence that minimizes the "current move sequence length + the number of misaligned pieces × α (approximately 4)" is selected as the output. Unlike cube, row shifts and 3-move shifts move large numbers of pieces at once and have less locality, so we only add moves to the end, not insert them at arbitrary times. 

Finally, we merged the move sequences determined for each cluster into a singular sequence, incorporating row shifts to nullify the offsets for each row. During this phase, we managed to reduce the number of moves by 1) sharing flip moves and 2) reversing row moves:

1. Sharing flip moves: If moves fj.r0.fj are executed on one row and fj.r1.fj on another, these can be amalgamated into fj.r0.r1.fj, resulting in a reduction of two moves.
2. Reversing row moves: By substituting r0 with -r1 and subsequently adjusting the index for fj applied to that row by one, the same state is achieved. This may enhance the likelihood of sharing flip moves and further diminishes the final offset, thereby reducing the number of moves required for offset correction.

While there is some flexibility in which flip moves are shared and which row moves are reversed, when dealing with two clusters, the number of columns being C, and the length of the move sequence being L, the optimal choice for merging can be calculated using a dynamic programming in O(C^2 L^2) time. When the number of clusters is three or more, we merge two clusters at a time with the DP.

## 4. Insert 3-rots at arbitrary times (cube)

In this step, we start from a state that is somewhat aligned and proceed to fully align it by inserting 3-rots at arbitrary time. We calculate the "minimum number of 3-rot insertions needed to fully align" as follows, and we only insert 3-rots that decreases this value.

For each cluster, we consider the cyclic permutation decomposition. When there are multiple pieces of the same color, the decomposition is not uniquely determined. However, if we denote the lengths of each cycle as (L1, L2, ..., Lk), then the minimum number of 3-rots is sum(floor(Li/2)). Hence, we aim to find a decomposition that minimizes this value. Since each cluster only has 24 pieces and is already somewhat aligned by step 3, this optimal cycle decomposition can be rapidly computed through exhaustive search.

We implemented two methods: a beam search that computes the top k shortest sequences for each remaining necessary number of 3-rots (with a beam width of about 100,000), and a one-step lookahead greedy method that inserts an additional 3-rots near the insertion time of the previous one. The beam search proved effective for smaller n, while the one-step lookahead greedy method was stronger for larger n.

## 4. Insert 3-rots at arbitrary times (globe)

Unlike cube, we could not compute the optimal cyclic permutation decomposition for large size problems because of the huge size of the clusters. Therefore, we used the "number of pieces that are not aligned" instead of the "number of 3-rots required" and performed a beam search (beam width of about 1000).
