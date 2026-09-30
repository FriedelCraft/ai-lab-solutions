% Direct connections, exactly as given in the optional laboratory.
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :- connected(X,Y).
valid_move(X,Y) :- connected(X,Y).

% Task 8 inference example.
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.

% Run with: swipl -q -s planner.pl -g run_checks -t halt
run_checks :-
    can_move(a,b), \+ can_move(a,c),
    valid_move(a,b), valid_move(b,c), \+ valid_move(a,c),
    reduce_speed,
    writeln('can_move(a,b): true'),
    writeln('can_move(a,c): false'),
    writeln('valid_move(a,b): true'),
    writeln('valid_move(b,c): true'),
    writeln('valid_move(a,c): false'),
    writeln('reduce_speed: true').
