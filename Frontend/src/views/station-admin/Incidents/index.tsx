import { useEffect, useState } from "react";
import TableSkeleton from "@/components/skeleton/TableSkeleton";
import IncidentTable from "./components/IncidentTable";
import { useAppSelector,useAppDispatch } from "@/app/store";
import { Incident } from "@/app/features/IncidentSlice";
import { SOURCES, getSource } from "../../../constants/validations";
import { fetchIncidents } from "@/app/features/IncidentSlice";
const Incidents = () => {

  const incidents = useAppSelector((state) => state.incidents.data);
  const [filteredIncidents, setFilteredIncidents] = useState<Incident[]>([]);
  const admin = useAppSelector((state) => state.admin.data);
  const dispatch = useAppDispatch();
  useEffect(() => {
    if (admin?.station_name) {
      // Initial fetch
      dispatch(fetchIncidents({ stationName: admin.station_name }));
      
      // Set up polling interval to fetch updated incidents
      const intervalId = setInterval(() => {
        dispatch(fetchIncidents({ stationName: admin.station_name }));
      }, 30000);
      
      // Clean up interval on unmount
      return () => clearInterval(intervalId);
    }
  }, []);
  useEffect(() => {
    incidents.forEach((obj: Incident) => {
      console.log('Object source:', obj.source);
      console.log('Get source result:', getSource(obj.source));
      console.log('Is CCTV?:', getSource(obj.source) === SOURCES.CCTV);
    });
    
    setFilteredIncidents(incidents);
  }, [incidents]);

  return (
    <div>
      <div className="mx-3 my-3 grid grid-cols-1">
        {filteredIncidents.length > 0 ? (
          (console.log(filteredIncidents),
          (<IncidentTable tableData={filteredIncidents} />))
        ) : (
          <TableSkeleton />
        )}
      </div>
    </div>
  );
};

export default Incidents;
