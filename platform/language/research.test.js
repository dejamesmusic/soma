import { commandToPath, pathToCommand } from './commands';

test('retired pages send old links to the notebook (Nov 25)', () => {
  expect(pathToCommand('/research')).toBe('developments');
  expect(pathToCommand('/lineage')).toBe('developments');
  expect(pathToCommand('/developments/soma-1b-a-new-frontier', '?version=1')).toBe('developments soma-1b-a-new-frontier?version=1');
});
