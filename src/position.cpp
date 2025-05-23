
#include "position.h"

#include "cuckoo.h"
#include "movegen.h"
#include "nnue/nn.h"
#include "precomputed.h"
#include "util.h"
#include "util/alloc.h"
#include "zobrist.h"

#include <cassert>
#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

namespace Horsie {

    Position::Position(const std::string& fen) {
        Hashes.reserve(MaxPly);

        LoadFromFEN(fen);
    }

    Move Position::TryFindMove(const std::string& moveStr, bool& found) const {
        Move move = Move::Null();
        found = false;
        
        auto eq_pred = [&](char a, char b) {
            return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
        };

        ScoredMove list[MoveListSize];
        i32 size = Generate<GenLegal>(*this, list, 0);
        for (i32 i = 0; i < size; i++) {
            Move m = list[i].move;

            std::string smith = m.SmithNotation(IsChess960);
            if (std::equal(smith.begin(), smith.end(), moveStr.begin(), moveStr.end(), eq_pred)) {
                found = true;
                return m;
            }

            std::string san = m.ToString(bb);
            if (std::equal(san.begin(), san.end(), moveStr.begin(), moveStr.end(), eq_pred)) {
                found = true;
                return m;
            }
        }

        std::cout << "No move '" << moveStr << "' found" << std::endl;
        for (i32 i = 0; i < size; i++) {
            Move m = list[i].move;
            std::cout << m.SmithNotation(IsChess960) << ", ";
        }
        std::cout << std::endl;
        for (i32 i = 0; i < size; i++) {
            Move m = list[i].move;
            std::cout << m.ToString(bb) << ", ";
        }
        std::cout << std::endl;

        return move;
    }

    void Position::MakeMove(Move move) {
        MakeMove<true>(move);
    }

    template<bool UpdateNN>
    void Position::MakeMove(Move move) {
        States.Add(State);
        Hashes.push_back(Hash());

        if (UpdateNN) {
            Accumulators.MakeMove(*this, move);
        }

        State.HalfmoveClock++;
        State.PliesFromNull++;

        if (ToMove == Color::BLACK) {
            FullMoves++;
        }

        const auto [moveFrom, moveTo] = move.Unpack();

        const i32 ourPiece = bb.GetPieceAtIndex(moveFrom);
        const i32 ourColor = ToMove;

        i32 theirPiece = bb.GetPieceAtIndex(moveTo);
        const i32 theirColor = Not(ourColor);

        assert(theirPiece != Piece::KING);
        assert(theirPiece == Piece::NONE || bb.GetColorAtIndex(moveTo) != ourColor || move.IsCastle());

        if (ourPiece == Piece::KING) {
            if (move.IsCastle()) {
                //  Move our rook and update the hash
                theirPiece = Piece::NONE;
                DoCastling(ourColor, moveFrom, moveTo, false);
                State.KingSquares[ourColor] = move.CastlingKingSquare();
            }
            else {
                State.KingSquares[ourColor] = moveTo;
            }

            RemoveCastling(ourColor == Color::WHITE ? CastlingStatus::White : CastlingStatus::Black);
        }
        else if (ourPiece == Piece::ROOK) {
            RemoveCastling(GetCastlingForRook(moveFrom));
        }

        State.CapturedPiece = theirPiece;
        if (theirPiece != Piece::NONE) {
            //  Remove their piece, and update the hash
            bb.RemovePiece(moveTo, theirColor, theirPiece);
            UpdateHash(theirColor, theirPiece, moveTo);

            if (theirPiece == Piece::ROOK) {
                RemoveCastling(GetCastlingForRook(moveTo));
            }

            //  Reset the halfmove clock
            State.HalfmoveClock = 0;
        }


        i32 tempEPSquare = State.EPSquare;
        if (State.EPSquare != EP_NONE) {
            //  Set st->EPSquare to 64 now.
            //  If we are capturing en passant, move.EnPassant is true. In any case it should be reset every move.
            Zobrist::EnPassant(State.Hash, GetIndexFile(State.EPSquare));
            State.EPSquare = EP_NONE;
        }

        if (ourPiece == Piece::PAWN) {
            if (move.IsEnPassant()) {
                i32 idxPawn = ((bb.Pieces[Piece::PAWN] & SquareBB(tempEPSquare - 8)) != 0) ? tempEPSquare - 8 : tempEPSquare + 8;

                bb.RemovePiece(idxPawn, theirColor, Piece::PAWN);
                UpdateHash(theirColor, Piece::PAWN, idxPawn);

                //  The EnPassant/Capture flags are mutually exclusive, so set CapturedPiece here
                State.CapturedPiece = Piece::PAWN;
            }
            else if ((moveTo ^ moveFrom) == 16) {
                i32 down = -ShiftUpDir(ourColor);

                //  st->EPSquare is only set if they have a pawn that can capture this one (via en passant)
                if ((PawnAttackMasks[ourColor][moveTo + down] & bb.Colors[theirColor] & bb.Pieces[PAWN]) != 0) {
                    State.EPSquare = moveTo + down;
                    Zobrist::EnPassant(State.Hash, GetIndexFile(State.EPSquare));
                }
            }

            //  Reset the halfmove clock
            State.HalfmoveClock = 0;
        }

        if (!move.IsCastle()) {
            bb.MoveSimple(moveFrom, moveTo, ourColor, ourPiece);

            UpdateHash(ourColor, ourPiece, moveFrom);
            UpdateHash(ourColor, ourPiece, moveTo);
        }

        if (move.IsPromotion()) {
            //  Get rid of the pawn we just put there
            bb.RemovePiece(moveTo, ourColor, ourPiece);

            //  And replace it with the promotion piece
            bb.AddPiece(moveTo, ourColor, move.PromotionTo());

            UpdateHash(ourColor, ourPiece, moveTo);
            UpdateHash(ourColor, move.PromotionTo(), moveTo);
        }

        Zobrist::ChangeToMove(State.Hash);
        ToMove = Not(ToMove);

        State.Checkers = bb.AttackersTo(State.KingSquares[theirColor], bb.Occupancy) & bb.Colors[ourColor];

        SetCheckInfo();
    }

    void Position::UnmakeMove(Move move) {
        UnmakeMove<true>(move);
    }

    template<bool UpdateNN>
    void Position::UnmakeMove(Move move) {
        const auto [moveFrom, moveTo] = move.Unpack();

        if (UpdateNN) {
            Accumulators.UndoMove();
        }

        //  Assume that "we" just made the last move, and "they" are undoing it.
        i32 ourPiece = bb.GetPieceAtIndex(moveTo);
        i32 ourColor = Not(ToMove);
        i32 theirColor = ToMove;

        if (move.IsPromotion()) {
            //  Remove the promotion piece and replace it with a pawn
            bb.RemovePiece(moveTo, ourColor, ourPiece);

            ourPiece = Piece::PAWN;

            bb.AddPiece(moveTo, ourColor, ourPiece);
        }
        else if (move.IsCastle()) {
            //  Put both pieces back
            DoCastling(ourColor, moveFrom, moveTo, true);
        }

        if (!move.IsCastle()) {
            //  Put our piece back to the square it came from.
            bb.MoveSimple(moveTo, moveFrom, ourColor, ourPiece);
        }

        if (State.CapturedPiece != Piece::NONE) {
            //  CapturedPiece is set for captures and en passant, so check which it was
            if (move.IsEnPassant()) {
                //  If the move was an en passant, put the captured pawn back

                i32 idxPawn = moveTo + ShiftUpDir(ToMove);
                bb.AddPiece(idxPawn, theirColor, Piece::PAWN);
            }
            else {
                //  Otherwise it was a capture, so put the captured piece back
                bb.AddPiece(moveTo, theirColor, State.CapturedPiece);
            }
        }

        if (ourColor == Color::BLACK) {
            //  If ourColor == Color.Black (and ToMove == White), then we incremented FullMoves when this move was made.
            FullMoves--;
        }

        State = States.RemoveLast();
        Hashes.pop_back();

        ToMove = Not(ToMove);
    }

    void Position::MakeNullMove() {
        States.Add(State);
        Hashes.push_back(Hash());

        if (State.EPSquare != EP_NONE) {
            //  Set EnPassantTarget to 64 now.
            //  If we are capturing en passant, move.EnPassant is true. In any case it should be reset every move.
            Zobrist::EnPassant(State.Hash, GetIndexFile(State.EPSquare));
            State.EPSquare = EP_NONE;
        }

        Zobrist::ChangeToMove(State.Hash);
        ToMove = Not(ToMove);
        State.HalfmoveClock++;
        State.PliesFromNull = 0;

        SetCheckInfo();
    }

    void Position::UnmakeNullMove() {
        State = States.RemoveLast();
        Hashes.pop_back();

        ToMove = Not(ToMove);
    }

    void Position::DoCastling(i32 ourColor, i32 from, i32 to, bool undo = false) {
        const bool kingSide = to > from;
        const auto rfrom = to;
        const auto rto = (kingSide ? static_cast<i32>(Square::F1) : static_cast<i32>(Square::D1)) ^ (ourColor * 56);
        const auto kto = (kingSide ? static_cast<i32>(Square::G1) : static_cast<i32>(Square::C1)) ^ (ourColor * 56);

        if (undo) {
            bb.RemovePiece(kto, ourColor, KING);
            bb.RemovePiece(rto, ourColor, ROOK);

            bb.AddPiece(from, ourColor, KING);
            bb.AddPiece(rfrom, ourColor, ROOK);
        }
        else {
            bb.RemovePiece(from, ourColor, KING);
            bb.RemovePiece(rfrom, ourColor, ROOK);

            bb.AddPiece(kto, ourColor, KING);
            bb.AddPiece(rto, ourColor, ROOK);

            UpdateHash(ourColor, KING, from);
            UpdateHash(ourColor, ROOK, rfrom);

            UpdateHash(ourColor, KING, kto);
            UpdateHash(ourColor, ROOK, rto);
        }
    }

    void Position::SetState() {
        State.Checkers = bb.AttackersTo(State.KingSquares[ToMove], bb.Occupancy) & bb.Colors[Not(ToMove)];

        SetCheckInfo();

        State.PawnHash = 0;
        State.NonPawnHash[WHITE] = State.NonPawnHash[BLACK] = 0;
        State.Hash = Zobrist::GetHash(*this, &State.PawnHash, &State.NonPawnHash[WHITE]);
        State.NonPawnHash[BLACK] = State.NonPawnHash[WHITE];
    }

    void Position::SetCheckInfo() {
        State.BlockingPieces[WHITE] = bb.BlockingPieces(WHITE, &State.Pinners[BLACK]);
        State.BlockingPieces[BLACK] = bb.BlockingPieces(BLACK, &State.Pinners[WHITE]);

        i32 kingSq = State.KingSquares[Not(ToMove)];

        State.CheckSquares[PAWN] = PawnAttackMasks[Not(ToMove)][kingSq];
        State.CheckSquares[HORSIE] = PseudoAttacks[HORSIE][kingSq];
        State.CheckSquares[BISHOP] = attacks_bb<BISHOP>(kingSq, bb.Occupancy);
        State.CheckSquares[ROOK] = attacks_bb<ROOK>(kingSq, bb.Occupancy);
        State.CheckSquares[QUEEN] = State.CheckSquares[BISHOP] | State.CheckSquares[ROOK];
        State.CheckSquares[KING] = 0;
    }

    void Position::SetCastlingStatus(i32 c, i32 rfrom) {
        const auto kfrom = bb.KingIndex(c);
        const auto cr = (c == WHITE && kfrom < rfrom) ? CastlingStatus::WK
                      : (c == BLACK && kfrom < rfrom) ? CastlingStatus::BK
                      : (c == WHITE)                  ? CastlingStatus::WQ
                      :                                 CastlingStatus::BQ;

        CastlingRookSquares[static_cast<i32>(cr)] = rfrom;

        auto kto = ((cr & CastlingStatus::Kingside) != CastlingStatus::None) ? static_cast<i32>(Square::G1) : static_cast<i32>(Square::C1);
        auto rto = ((cr & CastlingStatus::Kingside) != CastlingStatus::None) ? static_cast<i32>(Square::F1) : static_cast<i32>(Square::D1);

        kto ^= (56 * c);
        rto ^= (56 * c);

        CastlingRookPaths[static_cast<i32>(cr)] = (LineBB[rfrom][static_cast<i32>(rto)] | LineBB[kfrom][static_cast<i32>(kto)]) & ~(SquareBB(kfrom) | SquareBB(rfrom));

        State.CastleStatus |= cr;
    }

    u64 Position::HashAfter(Move move) const {
        u64 hash = State.Hash;

        const auto [moveFrom, moveTo] = move.Unpack();
        i32 us = bb.GetColorAtIndex(moveFrom);
        i32 ourPiece = bb.GetPieceAtIndex(moveFrom);

        if (bb.GetPieceAtIndex(moveTo) != Piece::NONE) {
            Zobrist::ToggleSquare(hash, Not(us), bb.GetPieceAtIndex(moveTo), moveTo);
        }

        Zobrist::Move(hash, moveFrom, moveTo, us, ourPiece);
        Zobrist::ChangeToMove(hash);

        return hash;
    }

    bool Position::IsPseudoLegal(Move move) const {
        const auto [moveFrom, moveTo] = move.Unpack();

        const auto pt = bb.GetPieceAtIndex(moveFrom);
        if (pt == Piece::NONE) {
            //  There isn't a piece on the move's "from" square.
            return false;
        }

        const auto pc = bb.GetColorAtIndex(moveFrom);
        if (pc != ToMove) {
            //  This isn't our piece, so we can't move it.
            return false;
        }

        if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && pc == bb.GetColorAtIndex(moveTo) && !move.IsCastle()) {
            //  There is a piece on the square we are moving to, and it is ours, so we can't capture it.
            //  The one exception is castling, which is encoded as king captures rook.
            return false;
        }

        if (pt == PAWN) {
            if (move.IsEnPassant()) {
                return State.EPSquare != EP_NONE && (SquareBB(moveTo - ShiftUpDir(ToMove)) & bb.Pieces[PAWN] & bb.Colors[Not(ToMove)]) != 0;
            }

            u64 empty = ~bb.Occupancy;
            if ((moveTo ^ moveFrom) != 16) {
                //  This is NOT a pawn double move, so it can only go to a square it attacks or the empty square directly above/below.
                return (bb.AttackMask(moveFrom, pc, pt, bb.Occupancy) & SquareBB(moveTo)) != 0
                    || (empty & SquareBB(moveTo)) != 0;
            }
            else {
                //  This IS a pawn double move, so it can only go to a square it attacks,
                //  or the empty square 2 ranks above/below provided the square 1 rank above/below is also empty.
                return (bb.AttackMask(moveFrom, pc, pt, bb.Occupancy) & SquareBB(moveTo)) != 0
                    || ((empty & SquareBB(moveTo - ShiftUpDir(pc))) != 0 && (empty & SquareBB(moveTo)) != 0);
            }

        }

        //  This move is only pseudo-legal if the piece that is moving is actually able to get there.
        //  Pieces can only move to squares that they attack, with the one exception of queenside castling
        return (bb.AttackMask(moveFrom, pc, pt, bb.Occupancy) & SquareBB(moveTo)) != 0 || move.IsCastle();
    }

    bool Position::IsLegal(Move move) const { return IsLegal(move, State.KingSquares[ToMove], State.KingSquares[Not(ToMove)], State.BlockingPieces[ToMove]); }

    bool Position::IsLegal(Move move, i32 ourKing, i32 theirKing, u64 pinnedPieces) const {
        const auto [moveFrom, moveTo] = move.Unpack();

        const auto pt = bb.GetPieceAtIndex(moveFrom);

        if (pt == NONE) {
            return false;
        }

        if (InDoubleCheck() && pt != Piece::KING) {
            //  Must move king out of double check
            return false;
        }

        const auto ourColor = ToMove;
        const auto theirColor = Not(ourColor);

        if (InCheck()) {
            //  We have 3 Options: block the check, take the piece giving check, or move our king out of it.

            if (pt == Piece::KING) {
                //  We need to move to a square that they don't attack.
                //  We also need to consider (NeighborsMask[moveTo] & SquareBB[theirKing]), because bb.AttackersTo does NOT include king attacks
                //  and we can't move to a square that their king attacks.
                return ((bb.AttackersTo(moveTo, bb.Occupancy ^ SquareBB(moveFrom)) & bb.Colors[theirColor]) | (PseudoAttacks[KING][moveTo] & SquareBB(theirKing))) == 0;
            }

            i32 checker = lsb(State.Checkers);
            if (((LineBB[ourKing][checker] & SquareBB(moveTo)) != 0)
                || (move.IsEnPassant() && GetIndexFile(moveTo) == GetIndexFile(checker))) {

                return pinnedPieces == 0 || (pinnedPieces & SquareBB(moveFrom)) == 0;
            }

            //  This isn't a king move and doesn't get us out of check, so it's illegal.
            return false;
        }

        if (pt == Piece::KING) {
            if (move.IsCastle()) {
                CastlingStatus thisCr = move.RelevantCastlingRight();
                i32 rookSq = CastlingRookSquare(thisCr);

                if ((SquareBB(rookSq) & bb.Pieces[ROOK] & bb.Colors[ourColor]) == 0) {
                    //  There isn't a rook on the square that we are trying to castle towards.
                    return false;
                }

                if (IsChess960 && (State.BlockingPieces[ourColor] & SquareBB(moveTo)) != 0) {
                    //  This rook was blocking a check, and it would put our king in check
                    //  once the rook and king switched places
                    return false;
                }

                i32 kingTo = (moveTo > moveFrom ? static_cast<i32>(Square::G1) : static_cast<i32>(Square::C1)) ^ (ourColor * 56);
                u64 them = bb.Colors[theirColor];
                i32 dir = (moveFrom < kingTo) ? -1 : 1;
                for (i32 sq = kingTo; sq != moveFrom; sq += dir) {
                    if ((bb.AttackersTo(sq, bb.Occupancy) & them) != 0) {
                        //  Moving here would put us in check
                        return false;
                    }
                }

                return ((bb.AttackersTo(kingTo, bb.Occupancy ^ SquareBB(ourKing)) & bb.Colors[theirColor])
                        | (PseudoAttacks[KING][kingTo] & SquareBB(theirKing))) == 0;
            }

            //  We can move anywhere as i64 as it isn't attacked by them.

            //  SquareBB[ourKing] is masked out from bb.Occupancy to prevent kings from being able to move backwards out of check,
            //  meaning a king on B1 in check from a rook on C1 can't actually go to A1.
            return ((bb.AttackersTo(moveTo, bb.Occupancy ^ SquareBB(ourKing)) & bb.Colors[theirColor])
                    | (PseudoAttacks[KING][moveTo] & SquareBB(theirKing))) == 0;
        }
        else if (move.IsEnPassant()) {
            //  En passant will remove both our pawn and the opponents pawn from the rank so this needs a special check
            //  to make sure it is still legal

            i32 idxPawn = moveTo - ShiftUpDir(ourColor);
            u64 moveMask = SquareBB(moveFrom) | SquareBB(moveTo);

            //  This is only legal if our king is NOT attacked after the EP is made
            return (bb.AttackersTo(ourKing, bb.Occupancy ^ (moveMask | SquareBB(idxPawn))) & bb.Colors[Not(ourColor)]) == 0;
        }

        //  Otherwise, this move is legal if:
        //  The piece we are moving isn't a blocker for our king
        //  The piece is a blocker for our king, but it is moving along the same ray that it had been blocking previously.
        //  (i.e. a rook on B1 moving to A1 to capture a rook that was pinning it to our king on C1)

        return ((State.BlockingPieces[ourColor] & SquareBB(moveFrom)) == 0) || ((RayBB[moveFrom][moveTo] & SquareBB(ourKing)) != 0);
    }

    bool Position::IsDraw(i16 ply) const {
        //  50 move rule, iff it's not checkmate (which takes priority)
        if (State.HalfmoveClock >= 100 && (!InCheck() || HasLegalMoves()))
            return true;

        //  Two repetitions before root or one after it
        const auto histSize = static_cast<i32>(Hashes.size());
        const auto dist = std::min(State.HalfmoveClock, histSize);
        bool rep = false;
        for (i32 i = 4; i <= dist; i += 2) {
            if (Hashes[histSize - i] == State.Hash) {
                if (ply >= i || rep)
                    return true;

                rep = true;
            }
        }

        //  Any pawns/rooks/queens are sufficient material
        if ((bb.Pieces[Piece::QUEEN] | bb.Pieces[Piece::ROOK] | bb.Pieces[Piece::PAWN]) != 0)
            return false;

        //  Otherwise, fewer than 2 minors is insufficient
        if (!MoreThanOne(bb.Pieces[Piece::HORSIE] | bb.Pieces[Piece::BISHOP]))
            return true;

        return false;
    }

    bool Position::HasLegalMoves() const {
        ScoredMove list[MoveListSize];
        i32 legals = Generate<GenLegal>(*this, list, 0);
        return legals != 0;
    }

    i32 Position::MaterialCount() const {
        return 9 * popcount(bb.Pieces[QUEEN]) +
               5 * popcount(bb.Pieces[ROOK]) +
               3 * popcount(bb.Pieces[BISHOP]) +
               3 * popcount(bb.Pieces[HORSIE]) +
               1 * popcount(bb.Pieces[PAWN]);
    }

    void Position::RemoveCastling(CastlingStatus cr) {
        Zobrist::Castle(State.Hash, State.CastleStatus, cr);
        State.CastleStatus &= ~cr;
    }

    constexpr CastlingStatus Position::GetCastlingForRook(i32 sq) const {
        CastlingStatus cr = sq == CastlingRookSquare(CastlingStatus::WQ) ? CastlingStatus::WQ
                          : sq == CastlingRookSquare(CastlingStatus::WK) ? CastlingStatus::WK
                          : sq == CastlingRookSquare(CastlingStatus::BQ) ? CastlingStatus::BQ
                          : sq == CastlingRookSquare(CastlingStatus::BK) ? CastlingStatus::BK
                          :                                                CastlingStatus::None;

        return cr;
    }

    void Position::UpdateHash(i32 pc, i32 pt, i32 sq) {
        Zobrist::ToggleSquare(State.Hash, pc, pt, sq);

        if (pt == PAWN)
            Zobrist::ToggleSquare(State.PawnHash, pc, pt, sq);
        else
            Zobrist::ToggleSquare(State.NonPawnHash[pc], pc, pt, sq);
    }


#define BULK_PERFT 1
    u64 Position::Perft(i32 depth) {
#if !defined(BULK_PERFT)
        if (depth == 0) {
            return 1;
        }
#endif

        ScoredMove movelist[MoveListSize];
        i32 size = Generate<GenLegal>(*this, &movelist[0], 0);

#if defined(BULK_PERFT)
        if (depth == 1) {
            return size;
        }
#endif

        u64 n = 0;
        for (i32 i = 0; i < size; i++) {
            Move m = movelist[i].move;

            MakeMove<false>(m);
            n += Perft(depth - 1);
            UnmakeMove<false>(m);
        }

        return n;
    }

    u64 Position::SplitPerft(i32 depth) {
        ScoredMove movelist[MoveListSize];
        ScoredMove* list = &movelist[0];
        i32 size = Generate<GenLegal>(*this, list, 0);

        u64 n, total = 0;
        for (i32 i = 0; i < size; i++) {
            Move m = list[i].move;

            MakeMove<false>(m);
            n = Perft(depth - 1);
            total += n;
            UnmakeMove<false>(m);

            std::cout << Move::ToString(m) << ": " << n << std::endl;
        }

        return total;
    }

    void Position::SetupForDFRC(i32 wIdx, i32 bIdx) {
        bb.Reset();

        for (i32 sq = static_cast<i32>(Square::A2); sq <= static_cast<i32>(Square::H2); sq++)
            bb.AddPiece(sq, WHITE, PAWN);

        for (i32 sq = static_cast<i32>(Square::A7); sq <= static_cast<i32>(Square::H7); sq++)
            bb.AddPiece(sq, BLACK, PAWN);

        IsChess960 = true;
        i32 wBackrank[8] = {};
        i32 bBackrank[8] = {};

        const std::pair<i32, i32> H5H[] = {
            {0, 0}, {0, 1}, {0, 2}, {0, 3},
            {1, 1}, {1, 2}, {1, 3},
            {2, 2}, {2, 3},
            {3, 3},
        };

        const auto FillWithScharnaglNumber = [&](i32 n, i32 types[8]) {
            const auto PlaceInSpot = [&](i32 pt, i32 skip) {
                i32 skips = 0;
                for (i32 i = 0; i < 8; i++) {
                    if (types[i] == 0 && skips++ >= skip) {
                        types[i] = pt;
                        break;
                    }
                }
            };

            const auto n2 = n / 4;
            const auto b1 = n % 4;

            const auto n3 = n2 / 4;
            const auto b2 = n2 % 4;

            const auto n4 = n3 / 6;
            const auto  q = n3 % 6;

            const auto [h1, h2] = H5H[n4];

            types[b1 * 2 + 1] = BISHOP;
            types[b2 * 2 + 0] = BISHOP;

            PlaceInSpot(QUEEN, q);

            PlaceInSpot(HORSIE, h1);
            PlaceInSpot(HORSIE, h2);

            PlaceInSpot(ROOK, 0);
            PlaceInSpot(KING, 0);
            PlaceInSpot(ROOK, 0);
        };

        FillWithScharnaglNumber(wIdx, wBackrank);
        FillWithScharnaglNumber(bIdx, bBackrank);

        for (i32 sq = 0; sq < 8; sq++) {
            bb.AddPiece(static_cast<i32>(Square::A1) + sq, WHITE, wBackrank[sq]);
            bb.AddPiece(static_cast<i32>(Square::A8) + sq, BLACK, bBackrank[sq]);

            if (wBackrank[sq] == KING)
                State.KingSquares[WHITE] = static_cast<i32>(Square::A1) + sq;

            if (bBackrank[sq] == KING)
                State.KingSquares[BLACK] = static_cast<i32>(Square::A8) + sq;
        }

        for (i32 sq = 0; sq < 8; sq++) {
            if (wBackrank[sq] == ROOK)
                SetCastlingStatus(WHITE, static_cast<i32>(Square::A1) + sq);

            if (bBackrank[sq] == ROOK)
                SetCastlingStatus(BLACK, static_cast<i32>(Square::A8) + sq);
        }

        LoadFromFEN(GetFEN());
    }

    void Position::LoadFromFEN(const std::string& fen) {
        bb.Reset();
        FullMoves = 1;

        Hashes.clear();
        States.Clear();
        State = {};

        unsigned char col, row, token;
        size_t idx;
        i32 sq = static_cast<i32>(Square::A8);
        std::istringstream ss(fen);
        ss >> std::noskipws;

        while ((ss >> token) && !isspace(token)) {
            if (isdigit(token))
                sq += (token - '0') * EAST;  // Advance the given number of files

            else if (token == '/')
                sq += 2 * SOUTH;

            else if ((idx = PieceToChar.find(tolower(token))) != std::string::npos) {
                i32 pc = isupper(token) ? WHITE : BLACK;
                i32 pt = Piece(idx);

                bb.AddPiece(sq, pc, pt);
                ++sq;
            }
        }
        ss >> token;
        ToMove = (token == 'w' ? WHITE : BLACK);
        ss >> token;
        while ((ss >> token) && !isspace(token)) {
            i32 rsq = SQUARE_NB;
            i32 color = isupper(token) ? WHITE : BLACK;
            char upper = toupper(token);

            if (upper == 'K') {
                for (rsq = (static_cast<i32>(Square::H1) ^ (56 * color)); bb.GetPieceAtIndex(rsq) != ROOK; --rsq) {}
            }
            else if (upper == 'Q') {
                for (rsq = (static_cast<i32>(Square::A1) ^ (56 * color)); bb.GetPieceAtIndex(rsq) != ROOK; ++rsq) {}
            }
            else if (upper >= 'A' && upper <= 'H') {
                IsChess960 = true;
                rsq = CoordToIndex(static_cast<i32>(upper) - 'A', (0 ^ (color * 7)));
            }
            else
                continue;

            SetCastlingStatus(color, rsq);
        }

        if (((ss >> col) && (col >= 'a' && col <= 'h')) && ((ss >> row) && (row == (ToMove == WHITE ? '6' : '3')))) {
            State.EPSquare = CoordToIndex(File(col - 'a'), Rank(row - '1'));
        }

        ss >> std::skipws >> State.HalfmoveClock >> FullMoves;

        State.KingSquares[WHITE] = bb.KingIndex(WHITE);
        State.KingSquares[BLACK] = bb.KingIndex(BLACK);

        SetState();

        Accumulators.Reset();
        Accumulators.RefreshIntoCache(*this);
        NNUE::ResetCaches(*this);
    }

    std::string Position::GetFEN() const {
        std::stringstream fen;

        for (i32 y = 7; y >= 0; y--) {
            i32 i = 0;
            for (i32 x = 0; x <= 7; x++) {
                i32 index = CoordToIndex(x, y);

                i32 pt = bb.GetPieceAtIndex(index);
                if (pt != Piece::NONE) {

                    if (i != 0) {
                        fen << std::to_string(i);
                        i = 0;
                    }

                    char c = PieceToFEN(pt);

                    if (bb.GetColorAtIndex(index) == WHITE) {
                        c = toupper(c);
                    }

                    fen << c;
                    continue;
                }
                else {
                    i++;
                }

                if (x == 7) {
                    fen << std::to_string(i);
                }
            }
            if (y != 0) {
                fen << "/";
            }
        }

        fen << (ToMove == Color::WHITE ? " w " : " b ");

        if (State.CastleStatus != CastlingStatus::None) {
            if ((State.CastleStatus & CastlingStatus::WK) != CastlingStatus::None) {
                fen << (IsChess960 ? (char)('A' + GetIndexFile(CastlingRookSquare(CastlingStatus::WK))) : 'K');
            }
            if ((State.CastleStatus & CastlingStatus::WQ) != CastlingStatus::None) {
                fen << (IsChess960 ? (char)('A' + GetIndexFile(CastlingRookSquare(CastlingStatus::WQ))) : 'Q');
            }
            if ((State.CastleStatus & CastlingStatus::BK) != CastlingStatus::None) {
                fen << (IsChess960 ? (char)('a' + GetIndexFile(CastlingRookSquare(CastlingStatus::BK))) : 'k');
            }
            if ((State.CastleStatus & CastlingStatus::BQ) != CastlingStatus::None) {
                fen << (IsChess960 ? (char)('a' + GetIndexFile(CastlingRookSquare(CastlingStatus::BQ))) : 'q');
            }
        }
        else {
            fen << "-";
        }

        if (State.EPSquare != EP_NONE) {
            fen << " " << IndexToString(State.EPSquare);
        }
        else {
            fen << " -";
        }

        fen << " " << std::to_string(State.HalfmoveClock) << " " << std::to_string(FullMoves);

        return fen.str();
    }

    std::ostream& operator<<(std::ostream& os, const Position& pos) {
        os << "\n +---+---+---+---+---+---+---+---+\n";

        for (Rank r = RANK_8; r >= RANK_1; --r) {
            for (File f = FILE_A; f <= FILE_H; ++f) {
                i32 sq = CoordToIndex(f, r);
                i32 pt = pos.bb.GetPieceAtIndex(sq);
                os << " | ";
                if (pt != Piece::NONE) {
                    char c = PieceToChar[pt];
                    os << (pos.bb.GetColorAtIndex(sq) == WHITE ? (char)toupper(c) : c);
                }
                else {
                    os << " ";
                }
            }

            os << " | " << (1 + r) << "\n +---+---+---+---+---+---+---+---+\n";
        }

        os << "   a   b   c   d   e   f   g   h\n";
        os << "\nFen: " << pos.GetFEN() << "\n";
        os << "\nHash: " << pos.Hash() << "\n";

        return os;
    }

    bool Position::SEE_GE(Move move, i32 threshold) const {
        if (move.IsCastle() || move.IsEnPassant() || move.IsPromotion()) {
            return threshold <= 0;
        }

        const auto [moveFrom, moveTo] = move.Unpack();

        i32 swap = GetSEEValue(bb.GetPieceAtIndex(moveTo)) - threshold;
        if (swap < 0)
            return false;

        swap = GetSEEValue(bb.GetPieceAtIndex(moveFrom)) - swap;
        if (swap <= 0)
            return true;

        u64 occ = (bb.Occupancy ^ SquareBB(moveFrom) | SquareBB(moveTo));

        u64 attackers = bb.AttackersTo(moveTo, occ);
        u64 stmAttackers;
        u64 temp;

        i32 stm = ToMove;
        i32 res = 1;
        while (true) {
            stm = Not(stm);
            attackers &= occ;

            stmAttackers = attackers & bb.Colors[stm];
            if (stmAttackers == 0) {
                break;
            }

            if ((State.Pinners[Not(stm)] & occ) != 0) {
                stmAttackers &= ~State.BlockingPieces[stm];
                if (stmAttackers == 0) {
                    break;
                }
            }

            res ^= 1;

            if ((temp = stmAttackers & bb.Pieces[PAWN]) != 0) {
                occ ^= SquareBB(lsb(temp));
                if ((swap = SEEValuePawn - swap) < res)
                    break;

                attackers |= GetBishopMoves(moveTo, occ) & (bb.Pieces[BISHOP] | bb.Pieces[QUEEN]);
            }
            else if ((temp = stmAttackers & bb.Pieces[HORSIE]) != 0) {
                occ ^= SquareBB(lsb(temp));
                if ((swap = SEEValueHorsie - swap) < res)
                    break;
            }
            else if ((temp = stmAttackers & bb.Pieces[BISHOP]) != 0) {
                occ ^= SquareBB(lsb(temp));
                if ((swap = SEEValueBishop - swap) < res)
                    break;

                attackers |= GetBishopMoves(moveTo, occ) & (bb.Pieces[BISHOP] | bb.Pieces[QUEEN]);
            }
            else if ((temp = stmAttackers & bb.Pieces[ROOK]) != 0) {
                occ ^= SquareBB(lsb(temp));
                if ((swap = SEEValueRook - swap) < res)
                    break;

                attackers |= GetRookMoves(moveTo, occ) & (bb.Pieces[ROOK] | bb.Pieces[QUEEN]);
            }
            else if ((temp = stmAttackers & bb.Pieces[QUEEN]) != 0) {
                occ ^= SquareBB(lsb(temp));
                if ((swap = SEEValueQueen - swap) < res)
                    break;

                attackers |= (GetBishopMoves(moveTo, occ) & (bb.Pieces[BISHOP] | bb.Pieces[QUEEN])) | (GetRookMoves(moveTo, occ) & (bb.Pieces[ROOK] | bb.Pieces[QUEEN]));
            }
            else {
                if ((attackers & ~bb.Pieces[stm]) != 0) {
                    return (res ^ 1) != 0;
                }
                else {
                    return res != 0;
                }
            }
        }

        return res != 0;
    }

    bool Position::HasCycle(i32 ply) const {
        using namespace Horsie::Cuckoo;

        const auto occ = bb.Occupancy;

        i32 dist = std::min(State.HalfmoveClock, State.PliesFromNull);
        if (dist < 3)
            return false;

        const auto HashFromStack = [&](i32 i) { return Hashes[Hashes.size() - i]; };

        auto other = State.Hash ^ HashFromStack(1) ^ Zobrist::BlackHash;

        i32 slot;
        for (i32 i = 3; i <= dist; i += 2) {
            const auto currKey = HashFromStack(i);
            other ^= currKey ^ HashFromStack(i - 1) ^ Zobrist::BlackHash;

            if (other != 0)
                continue;

            const auto diff = State.Hash ^ currKey;

            if (diff != keys[(slot = Hash1(diff))] &&
                diff != keys[(slot = Hash2(diff))])
                continue;

            Move move = moves[slot];
            const auto [moveFrom, moveTo] = move.Unpack();

            if (occ & BetweenBB[moveFrom][moveTo])
                continue;

            if (ply >= i)
                return true;

            for (int j = i + 4; j <= dist; j += 2) {
                if (HashFromStack(j) == HashFromStack(i))
                    return true;
            }
        }

        return false;
    }
}
