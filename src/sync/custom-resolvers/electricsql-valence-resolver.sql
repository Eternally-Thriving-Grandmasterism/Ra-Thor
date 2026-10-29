-- src/sync/custom-resolvers/electricsql-valence-resolver.sql
-- Valence-weighted resolver for critical columns (example: valence, experience)
-- Paste into ElectricSQL shape/schema migrations

CREATE RESOLVER valence_resolver FOR valence AS
  SELECT CASE
    WHEN NEW.valence > OLD.valence + 0.05 THEN NEW.valence
    WHEN OLD.valence > NEW.valence + 0.05 THEN OLD.valence
    ELSE GREATEST(NEW.valence, OLD.valence)
  END;

CREATE RESOLVER experience_resolver FOR experience AS
  SELECT CASE
    WHEN NEW.experience > OLD.experience THEN NEW.experience
    ELSE OLD.experience
  END;
