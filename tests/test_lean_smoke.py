import unittest

from src.evaluation.lean_smoke import proof_body, statement_prefix


class LeanSmokeTests(unittest.TestCase):
    def test_extract_tactic_body(self):
        self.assertEqual(proof_body("```lean\nby\n  ring\n```"), "by\n  ring")
        self.assertEqual(proof_body("by\n  exact Nat.add_comm a b"), "by\n  exact Nat.add_comm a b")

    def test_reject_bypass_constructs(self):
        for proof in ["by sorry", "by admit", "by native_decide", "by\n run_tac IO.println 1", "by\n exact True.intro\naxiom x : False", "by\n #eval 1", "by\n /- hidden -/ rfl", "by\n exact Lean.ofReduceBool"]:
            with self.subTest(proof=proof), self.assertRaises(ValueError):
                proof_body(proof)

    def test_declaration_not_accepted_as_proof(self):
        with self.assertRaises(ValueError):
            proof_body("theorem target : True := by trivial")

    def test_statement_preserves_goal(self):
        self.assertEqual(statement_prefix("theorem original (n : Nat) : n = n := sorry"), "theorem candidate_proof (n : Nat) : n = n := ")
        with self.assertRaises(ValueError):
            statement_prefix("theorem original : True := by trivial")


if __name__ == "__main__":
    unittest.main()
