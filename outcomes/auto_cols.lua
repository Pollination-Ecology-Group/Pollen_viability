function Table(tbl)
  for i = 1, #tbl.colspecs do
    tbl.colspecs[i] = {tbl.colspecs[i][1], nil}
  end
  return tbl
end

